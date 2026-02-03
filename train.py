from itertools import islice
import numpy as np
import torch
from torch import nn as nn
from torch.optim import lr_scheduler # [新增] 引入调度器

from evaluate import evaluate_model
from loss import AdversarialLoss, update_epoch_loss
from util import batch_generator, device


def prepare_labels(y_true, sequence_length, step_forward, forward_input_size, use_adversarial, pre_y_pred=None):
    if step_forward > 1:
        labels = y_true[:, sequence_length:, :]
    else:
        labels = y_true[:, sequence_length: sequence_length + 1, :]

    if not use_adversarial:
        return labels, None, None

    # 单步，多步训练都可以进行对抗
    if step_forward > 1:
        if forward_input_size > 0:
            pre_labels = y_true[:, sequence_length - 1: -1, :]
        else:
            pre_labels = y_true[:, sequence_length - 1: sequence_length, :]
        pred_pre_y = pre_y_pred[:, sequence_length - 1:, :]
    else:
        pre_labels = y_true[:, sequence_length - 1: sequence_length, :]
        pred_pre_y = pre_y_pred[:, sequence_length - 1:, :]

    return labels, pre_labels, pred_pre_y

# === [新增] TimeMixer 专用训练函数 (复现原文逻辑) ===
def _train_one_epoch_TimeMixer(model, train_data, criterion, optimizer, scheduler, args, epoch):
    model.train()
    epoch_loss = 0
    batch_count = 0
    
    # 模拟 data_loader
    batches = islice(
        batch_generator(
            data=train_data,
            batch_size=args.batch_size,
            past_input_size=args.past_input_size,
            forward_input_size=args.forward_input_size,
            shuffle=True
        ),
        args.batch_num
    )
    
    for past, forward, target, _, agent_ids, _, _, x_mark in batches:
        # 1. 数据准备
        x = torch.FloatTensor(past).to(device) if past is not None else None
        y_true = torch.FloatTensor(target).to(device)
        
        # TimeMixer 的 x_mark 处理
        if x_mark is not None:
            x_mark = torch.FloatTensor(x_mark).to(device)
            
        optimizer.zero_grad()
        
        # 2. 前向传播 (全量输出)
        # 注意：这里我们不需要 is_training=True，因为 TimeMixer 是非自回归的
        # 这里的 Adapter 应该配置为输出所有通道 (CI=0, c_out=enc_in)
        _, outputs = model(
            X=x, 
            y=None, 
            X_forward=None, 
            is_training=True, 
            x_mark=x_mark
        )
        
        # 3. [核心复现] 维度切片与对齐
        # 原文逻辑：f_dim = -1 if features=='MS' else 0
        # 你的任务是 MS (多变量预测单变量)，所以我们需要取最后一维
        
        # 截取预测长度 (Adapter可能已经做过，但再做一次无妨)
        pred = outputs[:, -args.step_forward:, :]
        true = y_true[:, -args.step_forward:, :]
        
        # [关键] 强制切片：只取 Target 维度计算 Loss
        # 假设 Target 是最后一个特征 (由 util.py 保证)
        pred = pred[:, :, -1:]
        true = true[:, :, -1:]
        
        # 4. 计算 Loss
        loss = criterion(pred, true)
        
        # 5. 反向传播
        loss.backward()
        
        # [可选] 梯度裁剪 (虽然 OneCycleLR 通常不需要，但加上更稳)
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        
        optimizer.step()
        
        # 6. [核心复现] 调度器 Step (OneCycleLR 是每个 Batch 更新一次)
        if scheduler is not None:
            scheduler.step()
            
        epoch_loss += loss.item()
        batch_count += 1
        
    return {'total': epoch_loss / batch_count if batch_count > 0 else 0, 'x':0, 'y':0}
    
def train_model(model, train_data, valid_data, scaler, args, use_adversarial=True):
    """
    完整的对抗训练函数

    参数:
        model: 要训练的模型
        train_data: 训练数据
        valid_data: 验证数据
        scaler: 数据标准化器
        args: 训练参数
        use_adversarial: 是否使用对抗训练

    返回:
        model: 训练好的模型
        history: 训练历史记录
    """
    # 初始化训练组件
    criterion, optimizers = _init_training_components(model, args, use_adversarial)
    history = _init_history()

    # === [新增] TimeMixer 专用配置 ===
    scheduler = None
    if args.model == 'TimeMixer':
        print(f"⚡ [TimeMixer Mode] 启用 OneCycleLR 调度器 (Max LR: {args.lr})")
        # 计算总步数: Epochs * Steps_per_epoch
        # 注意: batch_generator 是通过 islice(..., args.batch_num) 控制的
        steps_per_epoch = args.batch_num
        
        scheduler = lr_scheduler.OneCycleLR(
            optimizer=optimizers['default'],
            steps_per_epoch=steps_per_epoch,
            epochs=args.epochs,
            max_lr=args.lr, # 这里应该是 0.01 (从 yaml 读取)
            pct_start=0.3,  # 原文默认值，30% 时间热身
            div_factor=100, # 初始 LR = max_lr / 100
            final_div_factor=1000 # 最终 LR 极小
        )
    else:
        # === 你的模型专用温和策略 ===
        # 方案 A: 验证集 Loss 不降时自动减半 LR (最稳健)
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizers['default'] if not use_adversarial else optimizers['y'], 
            mode = 'min', factor = 0.8, patience = 5 # verbose=True
        )
        # 方案 B: 余弦退火 (无需验证集，自动衰减)
        # scheduler = lr_scheduler.CosineAnnealingLR(..., T_max=args.epochs)
        
    # 早停参数初始化
    best_loss = np.inf
    patience = args.patience
    no_improve_epochs = 0
    best_model_state = None
    for epoch in range(args.epochs):
        if args.model == 'TimeMixer':
            epoch_loss = _train_one_epoch_TimeMixer(
                model, train_data, criterion, optimizers['default'], 
                scheduler, args, epoch
            )
            # TimeMixer 不使用对抗训练逻辑，所有 loss 都在 'total' 里
            history['mode'].append('NOR') # 标记为普通训练
        else:
            # 原有逻辑
            epoch_loss = _train_one_epoch(
                model, train_data, criterion, optimizers,
                args, use_adversarial, epoch
            )
            # 原有记录逻辑
            is_adv = use_adversarial and (epoch % (args.normal_epochs + args.adv_epochs)) >= args.normal_epochs
            history['mode'].append('ADV' if is_adv else 'NOR')

        # 更新历史 (注意 TimeMixer 返回的结构可能不同，做兼容处理)
        history['total'].append(epoch_loss.get('total', 0))
        history['x'].append(epoch_loss.get('x', 0))
        history['y'].append(epoch_loss.get('y', 0))

        '''
        # 训练一个epoch
        epoch_loss = _train_one_epoch(
            model, train_data, criterion, optimizers,
            args, use_adversarial, epoch
        )

        # 记录训练历史
        _update_history(history, epoch_loss, epoch, args, use_adversarial)
        '''
        
        # 验证
        val_loss = _validate_model(model, valid_data, scaler, args)
        history['val_loss'].append(val_loss)

        # 3. Scheduler 更新
        if args.model == 'TimeMixer':
            pass # OneCycleLR 已经在 batch 内部 update 了
        else:
            # ReduceLROnPlateau 需要传入 val_loss
            if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
                
        # 早停逻辑
        """检查早停条件并更新最佳模型"""
        if val_loss < best_loss:
            best_loss = val_loss
            no_improve_epochs = 0
            best_model_state = model.state_dict()
        else:
            no_improve_epochs += 1

        # 打印日志
        # 确定使用哪个 optimizer 来查看 LR
        current_optim = None
        if 'default' in optimizers:
            current_optim = optimizers['default']
        elif 'y' in optimizers:
            current_optim = optimizers['y']  # 对抗模式下通常关注主任务优化器
        
        # 打印日志（传入找到的优化器）
        _print_epoch_log(epoch, args.epochs, history, val_loss, best_loss, optimizer=current_optim)


        if args.model == 'TimeMixer':
            # 打印当前 LR 确认调度器在工作
            current_lr = optimizers['default'].param_groups[0]['lr']
            # print(f"    LR: {current_lr:.6f}")

        
        # 检查是否早停
        if no_improve_epochs >= patience:
            print(f"Early stopping at epoch {epoch + 1} with best huber loss: {best_loss:.4f}")
            model.load_state_dict(best_model_state)
            break

    return model, history


def _init_training_components(model, args, use_adversarial, loss_fn = 'MSE'):
    """初始化损失函数和优化器"""
    # criterion = AdversarialLoss(args.adv_weight, args.nor_weight) if use_adversarial else nn.MSELoss()
    print('use_adversarial: ', use_adversarial)
    if loss_fn == 'MSE':
        criterion = AdversarialLoss(args.adv_weight, args.nor_weight, loss_fn) if use_adversarial else nn.MSELoss() # nn.L1Loss()
    elif loss_fn == 'huber': 
        criterion = AdversarialLoss(args.adv_weight, args.nor_weight, loss_fn) if use_adversarial else nn.HuberLoss(delta = 1.0) # nn.L1Loss()
    else:
        print('unknown loss_fn, set MSE')
        loss_fn = 'MSE'
        criterion = AdversarialLoss(args.adv_weight, args.nor_weight, loss_fn) if use_adversarial else nn.MSELoss() # nn.L1Loss()

    if use_adversarial:
        # 获取对抗部分的参数
        x_params = list(model.feature_tower.outNet.parameters())
        if hasattr(model.feature_tower, 'decoder_outNet'):
            x_params += list(model.feature_tower.decoder_outNet.parameters())

        # 获取主任务部分的参数
        y_params = [
            p for n, p in model.feature_tower.named_parameters()
            if not n.startswith('outNet.') and not n.startswith('decoder_outNet.')
        ]
        y_params += list(model.target_tower.parameters())

        optimizers = {
            'x': torch.optim.Adam(x_params, lr=args.lr_adv, weight_decay=args.weight_decay),
            'y': torch.optim.Adam(y_params, lr=args.lr, weight_decay=args.weight_decay)
        }
    else:
        optimizers = {'default': torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)}

    return criterion, optimizers


def _init_history():
    """初始化训练历史记录"""
    return {
        'total': [],  # 总损失
        'x': [],  # 对抗损失
        'y': [],  # 主任务总损失
        'mode': [],  # 训练模式
        'val_loss': []  # 验证集 huber
    }


def _train_one_epoch(model, train_data, criterion, optimizers, args, use_adversarial, epoch):
    """训练一个epoch"""
    model.train()
    is_adversarial = use_adversarial and (epoch % (args.normal_epochs + args.adv_epochs)) >= args.normal_epochs
    epoch_loss = {'total': 0, 'x': 0, 'y': 0}
    batch_count = 0

    batches = islice(
        batch_generator(
            data=train_data,
            batch_size=args.batch_size,
            past_input_size=args.past_input_size,
            forward_input_size=args.forward_input_size,
            shuffle=True
        ),
        args.batch_num
    )
    
    # print('len(batches[0]):', len(batches[0]))
    for past, forward, target, _, agent_ids, _, _, x_mark in batches:
        # 转换数据为PyTorch张量
        x = torch.FloatTensor(past).to(device) if past is not None else None
        x_forward = torch.FloatTensor(forward).to(device) if forward is not None else None
        y_true = torch.FloatTensor(target).to(device)
        if agent_ids is not None:
            agent_ids = torch.LongTensor(agent_ids).to(device)  # 添加 agent_ids 处理
        
        # 新增 x_mark 转换 (仅当不为 None 时)
        if x_mark is not None:
            x_mark = torch.FloatTensor(x_mark).to(device)

        # === [修改点 B] 构建动态参数字典 ===
        forward_kwargs = {
            'X': x,
            'y': y_true,
            'X_forward': x_forward,
            'ids': agent_ids,
            'is_training': True
        }
        
        # === [关键点] 仅对 TimeMixer 注入 x_mark ===
        # 这样保证了 LSTM 等旧模型不会收到意外参数
        if args.model == 'TimeMixer' and x_mark is not None:
            forward_kwargs['x_mark'] = x_mark

        # 调用模型 (使用 **kwargs 解包)
        pre_y_pred, outputs = model(**forward_kwargs)
        '''
        # 将agent_ids传递给模型
        pre_y_pred, outputs = model(
            X = x,
            y = y_true,
            X_forward = x_forward,
            ids = agent_ids,  # 添加 agent_ids 参数
            is_training = True
        )
        '''
        labels, pre_labels, pred_pre_y = prepare_labels(
            y_true=y_true,
            sequence_length=args.sequence_length,
            step_forward=args.step_forward,
            forward_input_size=args.forward_input_size,
            use_adversarial=use_adversarial,
            pre_y_pred=pre_y_pred
        )

        # 计算并反向传播损失
        if use_adversarial:
            loss_result = criterion.compute_losses(
                outputs=outputs,
                labels=labels,
                pre_labels=pre_labels,
                pred_pre_y=pred_pre_y,
                is_adversarial=is_adversarial
            )

            optim = optimizers['x' if loss_result['update_x'] else 'y']
            optim.zero_grad()
            loss_result['total_loss'].backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0) 
            
            optim.step()
            epoch_loss = update_epoch_loss(epoch_loss, loss_result)
        else:
            loss = criterion(outputs, labels)
            optimizers['default'].zero_grad()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0) 
            
            optimizers['default'].step()
            epoch_loss['total'] += loss.item()
            epoch_loss['y'] += loss.item()

        batch_count += 1

    # 计算平均损失
    for key in epoch_loss:
        epoch_loss[key] /= batch_count

    return epoch_loss


def _update_history(history, epoch_loss, epoch, args, use_adversarial):
    """更新训练历史记录"""
    # 兼容 TimeMixer 的字典结构
    if isinstance(epoch_loss, dict):
        history['total'].append(epoch_loss.get('total', 0))
        history['x'].append(epoch_loss.get('x', 0))
        history['y'].append(epoch_loss.get('y', 0))
    else:
        history['total'].append(epoch_loss) # Fallback

    # 确定并记录当前训练模式
    if args.model == 'TimeMixer':
        history['mode'].append('NOR')
    else:
        is_adversarial = use_adversarial and (epoch % (args.normal_epochs + args.adv_epochs)) >= args.normal_epochs
        history['mode'].append('ADV' if is_adversarial else 'NOR')


def _validate_model(model, valid_data, scaler, args):
    """在验证集上评估模型"""
    _, _, val_huber = evaluate_model(
        model=model,
        test_data=valid_data,
        sequence_length=args.sequence_length,
        step_forward=args.step_forward,
        scaler=scaler,
        batch_size=args.batch_size,
        past_input_size=args.past_input_size,
        forward_input_size=args.forward_input_size,
        mode='Valid',
        verbose=False
    )
    return val_huber


def _print_epoch_log(epoch, total_epochs, history, val_loss, best_loss, optimizer=None):
    """打印epoch日志"""
    # 简单防错
    mode = history['mode'][-1] if history['mode'] else 'NOR'
    total = history['total'][-1] if history['total'] else 0
    
    # 获取当前学习率
    current_lr = 0.0
    if optimizer:
        # 获取第一个参数组的学习率
        current_lr = optimizer.param_groups[0]['lr']
    
    if (epoch + 1) % 1 == 0 or epoch == total_epochs - 1:
        log = [
            f"Epoch {epoch + 1}/{total_epochs} [{mode}]",
            f"Total: {total:.8f}",
            f"Val loss: {val_loss:.4f}",
            f"Best loss: {best_loss:.4f}",
            f"LR: {current_lr:.6f}" # <--- 新增 LR 打印
        ]
        print(" | ".join(log))