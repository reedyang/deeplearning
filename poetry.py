import numpy as np
import keras
from keras import layers
import re
import os
from collections import Counter
import json
import glob
import random
import signal
import sys

# 创建训练输出目录
def create_training_output_dir():
    output_dir = "training_output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建训练输出目录: {output_dir}")
    return output_dir

# 加载本地古诗数据集
def load_poetry_data():
    poetry_dir = "poetry"
    if not os.path.exists(poetry_dir):
        raise FileNotFoundError(f"找不到古诗数据集目录: {poetry_dir}")
    
    # 获取目录下所有JSON文件
    json_files = glob.glob(os.path.join(poetry_dir, "*.json"))
    if not json_files:
        raise FileNotFoundError(f"在目录 {poetry_dir} 中找不到任何JSON文件")
    
    print(f"找到 {len(json_files)} 个JSON文件")
    return json_files

# 处理古诗数据并构建词汇表
def build_vocabulary(json_files, output_dir):
    char_count = Counter()
    total_poems = 0
    
    # 处理每个JSON文件构建词汇表
    for file_path in json_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            for poem in data:
                if "title" in poem and "paragraphs" in poem:
                    content = "".join(poem["paragraphs"])
                    # 清理内容：移除标点、空格等
                    content = re.sub(r"[，。？；：！、“”\s]", "", content)
                    if len(content) >= 12:  # 只保留足够长的诗
                        char_count.update(content)
                        total_poems += 1
        except Exception as e:
            print(f"处理文件 {os.path.basename(file_path)} 时出错: {e}")
    
    print(f"总共处理了 {total_poems} 首诗")
    
    # 创建词汇表
    vocab = sorted(char_count.keys())  # 排序以保证一致性
    vocab.insert(0, "<PAD>")  # 填充标记
    vocab.insert(1, "<START>")  # 开始标记
    vocab.insert(2, "<UNK>")   # 未知字符标记
    
    char_to_idx = {char: idx for idx, char in enumerate(vocab)}
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}
    
    # 保存词汇表
    vocab_path = os.path.join(output_dir, "vocabulary.json")
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump({
            "char_to_idx": char_to_idx,
            "idx_to_char": idx_to_char,
            "vocab": vocab
        }, f, ensure_ascii=False)
    print(f"词汇表已保存到: {vocab_path}")
    
    return char_to_idx, idx_to_char, vocab, total_poems

# 数据生成器
class PoetryDataGenerator(keras.utils.Sequence):
    def __init__(self, json_files, char_to_idx, batch_size=256, seq_length=20, shuffle=True, **kwargs):
        # 调用父类构造函数
        super().__init__(**kwargs)
        
        self.json_files = json_files
        self.char_to_idx = char_to_idx
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.shuffle = shuffle
        self.vocab_size = len(char_to_idx)
        
        # 预加载文件列表
        self.file_poems = {}
        for file_path in self.json_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                poems = []
                for poem in data:
                    if "title" in poem and "paragraphs" in poem:
                        title = poem["title"].strip()
                        content = "".join(poem["paragraphs"])
                        content = re.sub(r"[，。？；：！、“”\s]", "", content)
                        if len(content) >= 12:
                            poems.append((title, content))
                self.file_poems[file_path] = poems
            except Exception as e:
                print(f"加载文件 {os.path.basename(file_path)} 时出错: {e}")
        
        # 初始化索引
        self.indexes = []
        self.on_epoch_end()
    
    def on_epoch_end(self):
        # 重置索引
        self.indexes = []
        for file_path, poems in self.file_poems.items():
            for poem_idx in range(len(poems)):
                # 计算每首诗可以生成的序列数
                title, content = poems[poem_idx]
                sequence = [self.char_to_idx["<START>"]] 
                for char in content:
                    sequence.append(self.char_to_idx.get(char, self.char_to_idx["<UNK>"]))
                
                # 为每个序列位置添加索引
                num_sequences = len(sequence) - self.seq_length
                if num_sequences > 0:
                    for seq_idx in range(num_sequences):
                        self.indexes.append((file_path, poem_idx, seq_idx))
        
        if self.shuffle:
            random.shuffle(self.indexes)
    
    def __len__(self):
        # 每个epoch的批次数
        return int(np.ceil(len(self.indexes) / self.batch_size))
    
    def __getitem__(self, index):
        # 生成一个批次的数据
        batch_X = []
        batch_y = []
        
        # 获取当前批次的索引范围
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, len(self.indexes))
        
        for i in range(start_idx, end_idx):
            file_path, poem_idx, seq_idx = self.indexes[i]
            title, content = self.file_poems[file_path][poem_idx]
            
            # 添加开始标记
            sequence = [self.char_to_idx["<START>"]]
            for char in content:
                sequence.append(self.char_to_idx.get(char, self.char_to_idx["<UNK>"]))
            
            # 获取序列
            X_seq = sequence[seq_idx:seq_idx + self.seq_length]
            y_char = sequence[seq_idx + self.seq_length]
            
            batch_X.append(X_seq)
            batch_y.append(y_char)
        
        # 转换为numpy数组
        batch_X = np.array(batch_X)
        batch_y = np.array(batch_y)
        
        # one-hot编码目标
        batch_y = keras.utils.to_categorical(batch_y, num_classes=self.vocab_size)
        
        return batch_X, batch_y

# 构建LSTM模型
def build_model(vocab_size, output_dir, embedding_dim=128, lstm_units=256):
    print("构建模型...")
    model = keras.Sequential([
        layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=20),
        layers.LSTM(lstm_units, return_sequences=True),
        layers.LSTM(lstm_units),
        layers.Dense(lstm_units, activation='relu'),
        layers.Dense(vocab_size, activation='softmax')
    ])
    
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    
    # 显式构建模型
    model.build(input_shape=(None, 20))
    
    # 尝试保存模型结构图
    try:
        model_img_path = os.path.join(output_dir, "model_structure.png")
        # 使用 keras.utils.plot_model 需要安装 graphviz 和 pydot
        # 如果不可用，则跳过
        keras.utils.plot_model(model, to_file=model_img_path, show_shapes=True)
        print(f"模型结构图已保存到: {model_img_path}")
    except ImportError:
        print("警告: 无法保存模型结构图，需要安装pydot和graphviz")
    except Exception as e:
        print(f"警告: 保存模型结构图时出错: {e}")
    
    model.summary()
    return model

# 生成古诗
def generate_poetry(model, char_to_idx, idx_to_char, title, num_chars=28, temperature=0.8):
    # 标题处理
    title = re.sub(r"[，。？；：！、“”\s]", "", title)
    title = title[:5]  # 限制标题长度
    
    # 初始序列（开始标记 + 标题）
    input_seq = [char_to_idx["<START>"]]
    for char in title:
        input_seq.append(char_to_idx.get(char, char_to_idx["<UNK>"]))
    
    # 填充或截断序列
    if len(input_seq) < 20:
        input_seq = input_seq + [char_to_idx["<PAD>"]] * (20 - len(input_seq))
    else:
        input_seq = input_seq[:20]
    
    generated_poem = []
    
    for _ in range(num_chars):
        # 预测下一个字符
        x = np.array([input_seq])
        preds = model.predict(x, verbose=0)[0]
        # 应用温度参数调整多样性
        if temperature > 0:
            preds = np.log(preds + 1e-8) / temperature  # 防止log(0)
            exp_preds = np.exp(preds)
            preds = exp_preds / np.sum(exp_preds)
            preds = np.nan_to_num(preds)  # 替换NaN为0
            preds = np.clip(preds, 0, 1)  # 保证概率非负
            total = np.sum(preds)
            if not np.isfinite(total) or total == 0 or np.any(np.isnan(preds)) or np.all(preds == 0):
                next_idx = np.argmax(preds)
            else:
                preds /= total  # 再次归一化
                next_idx = np.random.choice(len(preds), p=preds)
        else:
            next_idx = np.argmax(preds)
        # 修复KeyError
        if next_idx not in idx_to_char:
            next_char = "<UNK>"
        else:
            next_char = idx_to_char[next_idx]
        generated_poem.append(next_char)
        
        # 更新输入序列
        input_seq = input_seq[1:] + [next_idx]
    
    # 将生成的字符组合成诗
    generated_poem = "".join(generated_poem)
    
    # 添加标点（每7字加逗号，每14字加句号换行，避免重复逗号）
    punctuated = []
    for i, char in enumerate(generated_poem):
        punctuated.append(char)
        pos = i + 1
        if pos % 14 == 0 and pos < len(generated_poem):
            punctuated.append("。\n")
        elif pos % 7 == 0 and pos < len(generated_poem):
            punctuated.append("，")
    # 结尾补句号
    if len(punctuated) > 0:
        if punctuated[-1] not in ["，", "。", "\n"]:
            punctuated.append("。")
        elif punctuated[-1] == "，":
            punctuated[-1] = "。"
    
    return title + "：\n" + "".join(punctuated)

# 保存训练状态
def save_training_state(epoch, history, model_path, output_dir):
    state_path = os.path.join(output_dir, "training_state.json")
    state = {
        "epoch": epoch,
        "history": history.history,
        "model_path": os.path.basename(model_path)  # 只保存文件名
    }
    with open(state_path, "w") as f:
        json.dump(state, f)
    print(f"训练状态已保存: {state_path} (epoch={epoch})")

# 加载训练状态
def load_training_state(output_dir):
    state_path = os.path.join(output_dir, "training_state.json")
    if os.path.exists(state_path):
        with open(state_path, "r") as f:
            state = json.load(f)
        return state
    return None

# 清理训练状态
def cleanup_training_state(output_dir):
    state_path = os.path.join(output_dir, "training_state.json")
    if os.path.exists(state_path):
        os.remove(state_path)
        print(f"训练状态已清理: {state_path}")

# 加载词汇表
def load_vocabulary(output_dir):
    vocab_path = os.path.join(output_dir, "vocabulary.json")
    if os.path.exists(vocab_path):
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab_data = json.load(f)
        char_to_idx = vocab_data["char_to_idx"]
        # 修复idx_to_char的key类型为int，防止生成全为<UNK>
        idx_to_char = {int(k): v for k, v in vocab_data["idx_to_char"].items()}
        vocab = vocab_data["vocab"]
        return char_to_idx, idx_to_char, vocab
    return None, None, None

# 中断信号处理
def handle_interrupt(signum, frame):
    print("\n检测到中断信号，正在保存训练状态...")
    # 注意：这里不能直接访问训练变量，所以只能设置标志
    global interrupted
    interrupted = True
    # 立即退出程序
    sys.exit(1)

# 注册中断信号处理
interrupted = False
signal.signal(signal.SIGINT, handle_interrupt)
signal.signal(signal.SIGTERM, handle_interrupt)

# 自定义回调函数用于检测中断
class InterruptCallback(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.interrupted = False
    
    def on_batch_end(self, batch, logs=None):
        global interrupted
        if interrupted:
            self.interrupted = True
            self.model.stop_training = True
            print("\n检测到中断信号，停止训练...")

# 保存状态回调
class SaveStateCallback(keras.callbacks.Callback):
    def __init__(self, model_path, output_dir):
        super().__init__()
        self.model_path = model_path
        self.output_dir = output_dir
    def on_epoch_end(self, epoch, logs=None):
        save_training_state(epoch, self.model.history, self.model_path, self.output_dir)

# 主程序
def main():
    global interrupted
    try:
        output_dir = create_training_output_dir()
        json_files = load_poetry_data()
        char_to_idx, idx_to_char, vocab = load_vocabulary(output_dir)
        if char_to_idx is None:
            char_to_idx, idx_to_char, vocab, total_poems = build_vocabulary(json_files, output_dir)
        else:
            print("加载现有词汇表")
            total_poems = "未知"
        vocab_size = len(vocab)
        print(f"词汇表大小: {vocab_size} 个字符")
        batch_size = 512
        train_generator = PoetryDataGenerator(
            json_files, char_to_idx, batch_size=batch_size, shuffle=True
        )
        print(f"总训练样本数: {len(train_generator.indexes)}")
        print(f"每个epoch的批次数: {len(train_generator)}")
        state = load_training_state(output_dir)
        model_path = os.path.join(output_dir, "poetry_generator.keras")
        total_epochs = 30
        steps_per_epoch = len(train_generator)
        # 优先判断state，保证中断后能恢复训练
        if state:
            last_epoch = state.get('epoch', 0)
            print(f"发现训练状态: epoch={last_epoch}")
            model_file = state["model_path"]
            model_path = os.path.join(output_dir, model_file)
            if os.path.exists(model_path):
                model = keras.models.load_model(model_path)
            else:
                print(f"警告: 找不到模型文件 {model_path}, 重新构建模型")
                model = build_model(vocab_size, output_dir)
            initial_epoch = last_epoch + 1
            history = keras.callbacks.History()
            history.history = state['history']
            if initial_epoch < total_epochs:
                print(f"恢复训练: epoch={initial_epoch} 到 {total_epochs}")
                checkpoint = keras.callbacks.ModelCheckpoint(
                    model_path, monitor='loss', save_best_only=True, mode='min', verbose=1
                )
                interrupt_callback = InterruptCallback()
                save_state_callback = SaveStateCallback(model_path, output_dir)
                callbacks = [checkpoint, interrupt_callback, save_state_callback, history]
                try:
                    model.fit(
                        train_generator,
                        steps_per_epoch=steps_per_epoch,
                        epochs=total_epochs,
                        initial_epoch=initial_epoch,
                        callbacks=callbacks,
                        verbose=1
                    )
                except KeyboardInterrupt:
                    print("训练被用户中断")
                    interrupted = True
                if interrupt_callback.interrupted or interrupted:
                    print("训练被中断，保存当前状态...")
                    current_epoch = initial_epoch + model._train_counter.epoch_index
                    save_training_state(current_epoch, history, model_path, output_dir)
                    return
                else:
                    print("训练完成!")
                    cleanup_training_state(output_dir)
                    history_path = os.path.join(output_dir, "training_history.json")
                    with open(history_path, "w") as f:
                        json.dump(history.history, f)
                    print(f"训练历史已保存到: {history_path}")
            print("训练已完成，进入交互模式...")
        elif not os.path.exists(model_path):
            print("未检测到模型文件，开始新模型训练...")
            model = build_model(vocab_size, output_dir)
            checkpoint = keras.callbacks.ModelCheckpoint(
                model_path, monitor='loss', save_best_only=True, mode='min', verbose=1
            )
            interrupt_callback = InterruptCallback()
            save_state_callback = SaveStateCallback(model_path, output_dir)
            history = keras.callbacks.History()
            callbacks = [checkpoint, interrupt_callback, save_state_callback, history]
            try:
                history = model.fit(
                    train_generator,
                    steps_per_epoch=steps_per_epoch,
                    epochs=total_epochs,
                    callbacks=callbacks,
                    verbose=1
                )
            except KeyboardInterrupt:
                print("训练被用户中断")
                interrupted = True
            # 修正：保存当前已完成的epoch
            if interrupt_callback.interrupted or interrupted:
                print("训练被中断，保存当前状态...")
                # 获取已完成的epoch数和history
                if history and hasattr(history, 'epoch') and history.epoch:
                    current_epoch = history.epoch[-1]
                    hist_data = history
                elif hasattr(model, 'history') and hasattr(model.history, 'epoch') and model.history.epoch:
                    current_epoch = model.history.epoch[-1]
                    hist_data = model.history
                else:
                    current_epoch = 0
                    class Dummy:
                        history = {}
                    hist_data = Dummy()
                save_training_state(current_epoch, hist_data, model_path, output_dir)
                return
            print("训练完成!")
            cleanup_training_state(output_dir)
            history_path = os.path.join(output_dir, "training_history.json")
            with open(history_path, "w") as f:
                json.dump(history.history, f)
            print(f"训练历史已保存到: {history_path}")
            print("训练已完成，进入交互模式...")
        else:
            print(f"检测到已有模型文件 {model_path}，直接加载模型并跳过训练。")
            model = keras.models.load_model(model_path)
            print("模型已加载，进入交互模式...")
        # 统一交互模式
        while True:
            if interrupted:
                print("训练被中断，进入交互模式...")
                interrupted = False
            title = input("\n请输入古诗标题（输入'退出'结束）: ").strip()
            if title == "退出":
                break
            try:
                temperature = float(input("请输入创造性参数(0.1-1.0, 推荐0.8): ") or "0.8")
                poem = generate_poetry(model, char_to_idx, idx_to_char, title, temperature=temperature)
                print("\n生成的诗:")
                print(poem)
                output_file = os.path.join(output_dir, f"generated_poem_{title}.txt")
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(poem)
                print(f"生成的诗已保存到: {output_file}")
            except ValueError:
                print("错误：请输入有效的数字作为创造性参数")
    except FileNotFoundError as e:
        print(f"错误: {e}")
    except Exception as e:
        print(f"程序运行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()