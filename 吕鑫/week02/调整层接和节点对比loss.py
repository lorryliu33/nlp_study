import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ... (Data loading and preprocessing remains the same) ...
# 加载数据集
dataset = pd.read_csv("../../task_01/dataset.csv", sep="\t", header=None)
# 提取文本和标签,并转为list
texts = dataset[0].tolist()
string_labels = dataset[1].tolist()
# 将字符串标签转换为数字标签（模型只能处理数字）
label_to_index = {label: i for i, label in enumerate(set(string_labels))}
numerical_labels = [label_to_index[label] for label in string_labels]

char_to_index = {'<pad>': 0}
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)

index_to_char = {i: char for char, i in char_to_index.items()}
vocab_size = len(char_to_index)

max_len = 40


class CharBoWDataset(Dataset):
    def __init__(self, texts, labels, char_to_index, max_len, vocab_size):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.char_to_index = char_to_index
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.bow_vectors = self._create_bow_vectors()

    def _create_bow_vectors(self):
        tokenized_texts = []
        for text in self.texts:
            tokenized = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
            tokenized += [0] * (self.max_len - len(tokenized))
            tokenized_texts.append(tokenized)

        bow_vectors = []
        for text_indices in tokenized_texts:
            bow_vector = torch.zeros(self.vocab_size)
            for index in text_indices:
                if index != 0:
                    bow_vector[index] += 1
            bow_vectors.append(bow_vector)
        return torch.stack(bow_vectors)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.bow_vectors[idx], self.labels[idx]


class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim): # 层的个数 和 验证集精度
        # 层初始化
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()  # 添加激活函数，否则无用
        self.fc3 = nn.Linear(hidden_dim, output_dim)  # 第三层

    def forward(self, x):
        # 手动实现每层的计算
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out


char_dataset = CharBoWDataset(texts, numerical_labels, char_to_index, max_len, vocab_size) # 读取单个样本
dataloader = DataLoader(char_dataset, batch_size=32, shuffle=True) # 读取批量数据集 -》 batch数据

# hidden_dim = 128
hidden_dim = 64
# hidden_dim = 256
output_dim = len(label_to_index)
model = SimpleClassifier(vocab_size, hidden_dim, output_dim) # 维度和精度有什么关系？
criterion = nn.CrossEntropyLoss() # 损失函数 内部自带激活函数，softmax
optimizer = optim.SGD(model.parameters(), lr=0.01)

# epoch： 将数据集整体迭代训练一次
# batch： 数据集汇总为一批训练一次

num_epochs = 10
for epoch in range(num_epochs): # 12000， batch size 100 -》 batch 个数： 12000 / 100
    model.train()
    running_loss = 0.0
    for idx, (inputs, labels) in enumerate(dataloader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if idx % 50 == 0:
            print(f"Batch 个数 {idx}, 当前Batch Loss: {loss.item()}")


    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")


def classify_text(text, model, char_to_index, vocab_size, max_len, index_to_label):
    tokenized = [char_to_index.get(char, 0) for char in text[:max_len]]
    tokenized += [0] * (max_len - len(tokenized))

    bow_vector = torch.zeros(vocab_size)
    for index in tokenized:
        if index != 0:
            bow_vector[index] += 1

    bow_vector = bow_vector.unsqueeze(0)

    model.eval()
    with torch.no_grad():
        output = model(bow_vector)

    _, predicted_index = torch.max(output, 1)
    predicted_index = predicted_index.item()
    predicted_label = index_to_label[predicted_index]

    return predicted_label


index_to_label = {i: label for label, i in label_to_index.items()}

new_text = "帮我导航到北京"
predicted_class = classify_text(new_text, model, char_to_index, vocab_size, max_len, index_to_label)
print(f"输入 '{new_text}' 预测为: '{predicted_class}'")

new_text_2 = "查询明天北京的天气"
predicted_class_2 = classify_text(new_text_2, model, char_to_index, vocab_size, max_len, index_to_label)
print(f"输入 '{new_text_2}' 预测为: '{predicted_class_2}'")

# -----------------------------------------------------两层  128
# Batch 个数 0, 当前Batch Loss: 2.4822170734405518
# Batch 个数 50, 当前Batch Loss: 2.466000556945801
# Batch 个数 100, 当前Batch Loss: 2.4409120082855225
# Batch 个数 150, 当前Batch Loss: 2.4167938232421875
# Batch 个数 200, 当前Batch Loss: 2.4272515773773193
# Batch 个数 250, 当前Batch Loss: 2.366985321044922
# Batch 个数 300, 当前Batch Loss: 2.4035801887512207
# Batch 个数 350, 当前Batch Loss: 2.3371894359588623
# Epoch [1/10], Loss: 2.4143
# Batch 个数 0, 当前Batch Loss: 2.361220598220825
# Batch 个数 50, 当前Batch Loss: 2.3521034717559814
# Batch 个数 100, 当前Batch Loss: 2.277466297149658
# Batch 个数 150, 当前Batch Loss: 2.2924644947052
# Batch 个数 200, 当前Batch Loss: 2.2193198204040527
# Batch 个数 250, 当前Batch Loss: 2.2038910388946533
# Batch 个数 300, 当前Batch Loss: 2.1661362648010254
# Batch 个数 350, 当前Batch Loss: 2.1171555519104004
# Epoch [2/10], Loss: 2.2379
# Batch 个数 0, 当前Batch Loss: 2.031489849090576
# Batch 个数 50, 当前Batch Loss: 2.1142141819000244
# Batch 个数 100, 当前Batch Loss: 1.9650574922561646
# Batch 个数 150, 当前Batch Loss: 1.912542700767517
# Batch 个数 200, 当前Batch Loss: 2.0349042415618896
# Batch 个数 250, 当前Batch Loss: 1.8395906686782837
# Batch 个数 300, 当前Batch Loss: 1.6692687273025513
# Batch 个数 350, 当前Batch Loss: 1.9137076139450073
# Epoch [3/10], Loss: 1.9474
# Batch 个数 0, 当前Batch Loss: 1.718170166015625
# Batch 个数 50, 当前Batch Loss: 1.5076268911361694
# Batch 个数 100, 当前Batch Loss: 1.724724292755127
# Batch 个数 150, 当前Batch Loss: 1.74276602268219
# Batch 个数 200, 当前Batch Loss: 1.5398951768875122
# Batch 个数 250, 当前Batch Loss: 1.5289400815963745
# Batch 个数 300, 当前Batch Loss: 1.5669795274734497
# Batch 个数 350, 当前Batch Loss: 1.3648427724838257
# Epoch [4/10], Loss: 1.5692
# Batch 个数 0, 当前Batch Loss: 1.5257313251495361
# Batch 个数 50, 当前Batch Loss: 1.423504114151001
# Batch 个数 100, 当前Batch Loss: 1.351486325263977
# Batch 个数 150, 当前Batch Loss: 1.3500961065292358
# Batch 个数 200, 当前Batch Loss: 1.1378332376480103
# Batch 个数 250, 当前Batch Loss: 1.1218689680099487
# Batch 个数 300, 当前Batch Loss: 1.2437481880187988
# Batch 个数 350, 当前Batch Loss: 1.0835661888122559
# Epoch [5/10], Loss: 1.2309
# Batch 个数 0, 当前Batch Loss: 1.2013500928878784
# Batch 个数 50, 当前Batch Loss: 0.9035167694091797
# Batch 个数 100, 当前Batch Loss: 0.9283719062805176
# Batch 个数 150, 当前Batch Loss: 0.7570436596870422
# Batch 个数 200, 当前Batch Loss: 0.9330022931098938
# Batch 个数 250, 当前Batch Loss: 0.7426336407661438
# Batch 个数 300, 当前Batch Loss: 0.9290605783462524
# Batch 个数 350, 当前Batch Loss: 0.7899795174598694
# Epoch [6/10], Loss: 0.9891
# Batch 个数 0, 当前Batch Loss: 0.8975003957748413
# Batch 个数 50, 当前Batch Loss: 0.6134164333343506
# Batch 个数 100, 当前Batch Loss: 0.7407364249229431
# Batch 个数 150, 当前Batch Loss: 0.9039138555526733
# Batch 个数 200, 当前Batch Loss: 0.6693668961524963
# Batch 个数 250, 当前Batch Loss: 0.8911571502685547
# Batch 个数 300, 当前Batch Loss: 0.6195366382598877
# Batch 个数 350, 当前Batch Loss: 0.854444682598114
# Epoch [7/10], Loss: 0.8278
# Batch 个数 0, 当前Batch Loss: 0.9494524002075195
# Batch 个数 50, 当前Batch Loss: 1.0926449298858643
# Batch 个数 100, 当前Batch Loss: 0.8163652420043945
# Batch 个数 150, 当前Batch Loss: 0.9505890607833862
# Batch 个数 200, 当前Batch Loss: 0.8854873776435852
# Batch 个数 250, 当前Batch Loss: 0.7188058495521545
# Batch 个数 300, 当前Batch Loss: 0.6166806221008301
# Batch 个数 350, 当前Batch Loss: 0.507116436958313
# Epoch [8/10], Loss: 0.7227
# Batch 个数 0, 当前Batch Loss: 0.5722290873527527
# Batch 个数 50, 当前Batch Loss: 0.5679795742034912
# Batch 个数 100, 当前Batch Loss: 0.48937368392944336
# Batch 个数 150, 当前Batch Loss: 0.6530099511146545
# Batch 个数 200, 当前Batch Loss: 0.6632325053215027
# Batch 个数 250, 当前Batch Loss: 0.5689074397087097
# Batch 个数 300, 当前Batch Loss: 0.5234953165054321
# Batch 个数 350, 当前Batch Loss: 0.5108585357666016
# Epoch [9/10], Loss: 0.6466
# Batch 个数 0, 当前Batch Loss: 0.5466612577438354
# Batch 个数 50, 当前Batch Loss: 0.4875358045101166
# Batch 个数 100, 当前Batch Loss: 0.5791579484939575
# Batch 个数 150, 当前Batch Loss: 0.6683869361877441
# Batch 个数 200, 当前Batch Loss: 0.6163793802261353
# Batch 个数 250, 当前Batch Loss: 0.5841099619865417
# Batch 个数 300, 当前Batch Loss: 0.6230998635292053
# Batch 个数 350, 当前Batch Loss: 0.5108715295791626
# Epoch [10/10], Loss: 0.5882
# 输入 '帮我导航到北京' 预测为: 'Travel-Query'
# 输入 '查询明天北京的天气' 预测为: 'Weather-Query'
# ------------------------------------三层 128
# Batch 个数 0, 当前Batch Loss: 2.4903054237365723
# Batch 个数 50, 当前Batch Loss: 2.4850852489471436
# Batch 个数 100, 当前Batch Loss: 2.454423189163208
# Batch 个数 150, 当前Batch Loss: 2.437962293624878
# Batch 个数 200, 当前Batch Loss: 2.4317779541015625
# Batch 个数 250, 当前Batch Loss: 2.409919023513794
# Batch 个数 300, 当前Batch Loss: 2.4389142990112305
# Batch 个数 350, 当前Batch Loss: 2.4066245555877686
# Epoch [1/10], Loss: 2.4425
# Batch 个数 0, 当前Batch Loss: 2.4055874347686768
# Batch 个数 50, 当前Batch Loss: 2.392990827560425
# Batch 个数 100, 当前Batch Loss: 2.364823579788208
# Batch 个数 150, 当前Batch Loss: 2.3712642192840576
# Batch 个数 200, 当前Batch Loss: 2.413137912750244
# Batch 个数 250, 当前Batch Loss: 2.397721290588379
# Batch 个数 300, 当前Batch Loss: 2.3519656658172607
# Batch 个数 350, 当前Batch Loss: 2.394148349761963
# Epoch [2/10], Loss: 2.3760
# Batch 个数 0, 当前Batch Loss: 2.322939872741699
# Batch 个数 50, 当前Batch Loss: 2.4261457920074463
# Batch 个数 100, 当前Batch Loss: 2.291391611099243
# Batch 个数 150, 当前Batch Loss: 2.3379099369049072
# Batch 个数 200, 当前Batch Loss: 2.2629356384277344
# Batch 个数 250, 当前Batch Loss: 2.2195448875427246
# Batch 个数 300, 当前Batch Loss: 2.223175048828125
# Batch 个数 350, 当前Batch Loss: 2.2622201442718506
# Epoch [3/10], Loss: 2.3051
# Batch 个数 0, 当前Batch Loss: 2.2610018253326416
# Batch 个数 50, 当前Batch Loss: 2.230321168899536
# Batch 个数 100, 当前Batch Loss: 2.2061750888824463
# Batch 个数 150, 当前Batch Loss: 2.1769959926605225
# Batch 个数 200, 当前Batch Loss: 2.2000136375427246
# Batch 个数 250, 当前Batch Loss: 2.2121849060058594
# Batch 个数 300, 当前Batch Loss: 2.097856044769287
# Batch 个数 350, 当前Batch Loss: 2.0985989570617676
# Epoch [4/10], Loss: 2.1765
# Batch 个数 0, 当前Batch Loss: 2.0784127712249756
# Batch 个数 50, 当前Batch Loss: 2.0321006774902344
# Batch 个数 100, 当前Batch Loss: 1.879571557044983
# Batch 个数 150, 当前Batch Loss: 2.0569980144500732
# Batch 个数 200, 当前Batch Loss: 1.9119994640350342
# Batch 个数 250, 当前Batch Loss: 1.7713212966918945
# Batch 个数 300, 当前Batch Loss: 1.8779191970825195
# Batch 个数 350, 当前Batch Loss: 1.6241891384124756
# Epoch [5/10], Loss: 1.8894
# Batch 个数 0, 当前Batch Loss: 1.5308541059494019
# Batch 个数 50, 当前Batch Loss: 1.6515388488769531
# Batch 个数 100, 当前Batch Loss: 1.4335949420928955
# Batch 个数 150, 当前Batch Loss: 1.5432935953140259
# Batch 个数 200, 当前Batch Loss: 1.3847962617874146
# Batch 个数 250, 当前Batch Loss: 1.2561426162719727
# Batch 个数 300, 当前Batch Loss: 1.197786808013916
# Batch 个数 350, 当前Batch Loss: 1.607709527015686
# Epoch [6/10], Loss: 1.4858
# Batch 个数 0, 当前Batch Loss: 1.3987585306167603
# Batch 个数 50, 当前Batch Loss: 1.3411331176757812
# Batch 个数 100, 当前Batch Loss: 1.331946849822998
# Batch 个数 150, 当前Batch Loss: 0.8871480822563171
# Batch 个数 200, 当前Batch Loss: 1.0473519563674927
# Batch 个数 250, 当前Batch Loss: 1.0665448904037476
# Batch 个数 300, 当前Batch Loss: 1.0186588764190674
# Batch 个数 350, 当前Batch Loss: 0.7169075608253479
# Epoch [7/10], Loss: 1.1306
# Batch 个数 0, 当前Batch Loss: 1.1544318199157715
# Batch 个数 50, 当前Batch Loss: 1.1870959997177124
# Batch 个数 100, 当前Batch Loss: 0.9288825392723083
# Batch 个数 150, 当前Batch Loss: 1.0144671201705933
# Batch 个数 200, 当前Batch Loss: 0.5954219698905945
# Batch 个数 250, 当前Batch Loss: 0.6508578658103943
# Batch 个数 300, 当前Batch Loss: 0.5991900563240051
# Batch 个数 350, 当前Batch Loss: 0.7611802816390991
# Epoch [8/10], Loss: 0.8737
# Batch 个数 0, 当前Batch Loss: 0.9825116991996765
# Batch 个数 50, 当前Batch Loss: 0.5821513533592224
# Batch 个数 100, 当前Batch Loss: 0.5803787112236023
# Batch 个数 150, 当前Batch Loss: 0.8137922286987305
# Batch 个数 200, 当前Batch Loss: 0.5907501578330994
# Batch 个数 250, 当前Batch Loss: 0.4970499575138092
# Batch 个数 300, 当前Batch Loss: 0.7859436273574829
# Batch 个数 350, 当前Batch Loss: 0.7116890549659729
# Epoch [9/10], Loss: 0.7082
# Batch 个数 0, 当前Batch Loss: 0.5774505138397217
# Batch 个数 50, 当前Batch Loss: 0.636289656162262
# Batch 个数 100, 当前Batch Loss: 0.8575273752212524
# Batch 个数 150, 当前Batch Loss: 0.6141330003738403
# Batch 个数 200, 当前Batch Loss: 0.4305588901042938
# Batch 个数 250, 当前Batch Loss: 0.828802227973938
# Batch 个数 300, 当前Batch Loss: 0.5374228358268738
# Batch 个数 350, 当前Batch Loss: 0.6312077045440674
# Epoch [10/10], Loss: 0.6027
# 输入 '帮我导航到北京' 预测为: 'Travel-Query'
# 输入 '查询明天北京的天气' 预测为: 'Weather-Query'
# --------------------------------------------三层  64
# Batch 个数 0, 当前Batch Loss: 2.4995875358581543
# Batch 个数 50, 当前Batch Loss: 2.4918107986450195
# Batch 个数 100, 当前Batch Loss: 2.476008176803589
# Batch 个数 150, 当前Batch Loss: 2.467363119125366
# Batch 个数 200, 当前Batch Loss: 2.435802459716797
# Batch 个数 250, 当前Batch Loss: 2.4281578063964844
# Batch 个数 300, 当前Batch Loss: 2.4496936798095703
# Batch 个数 350, 当前Batch Loss: 2.419644355773926
# Epoch [1/10], Loss: 2.4511
# Batch 个数 0, 当前Batch Loss: 2.393146514892578
# Batch 个数 50, 当前Batch Loss: 2.415121078491211
# Batch 个数 100, 当前Batch Loss: 2.407884359359741
# Batch 个数 150, 当前Batch Loss: 2.4002721309661865
# Batch 个数 200, 当前Batch Loss: 2.3895740509033203
# Batch 个数 250, 当前Batch Loss: 2.3286001682281494
# Batch 个数 300, 当前Batch Loss: 2.3600897789001465
# Batch 个数 350, 当前Batch Loss: 2.361325740814209
# Epoch [2/10], Loss: 2.3790
# Batch 个数 0, 当前Batch Loss: 2.376150131225586
# Batch 个数 50, 当前Batch Loss: 2.2843661308288574
# Batch 个数 100, 当前Batch Loss: 2.2730650901794434
# Batch 个数 150, 当前Batch Loss: 2.362525701522827
# Batch 个数 200, 当前Batch Loss: 2.293982744216919
# Batch 个数 250, 当前Batch Loss: 2.282412528991699
# Batch 个数 300, 当前Batch Loss: 2.2059531211853027
# Batch 个数 350, 当前Batch Loss: 2.1845436096191406
# Epoch [3/10], Loss: 2.2984
# Batch 个数 0, 当前Batch Loss: 2.1794960498809814
# Batch 个数 50, 当前Batch Loss: 2.2972073554992676
# Batch 个数 100, 当前Batch Loss: 2.1764471530914307
# Batch 个数 150, 当前Batch Loss: 2.1578829288482666
# Batch 个数 200, 当前Batch Loss: 2.2016115188598633
# Batch 个数 250, 当前Batch Loss: 2.2146480083465576
# Batch 个数 300, 当前Batch Loss: 2.038059949874878
# Batch 个数 350, 当前Batch Loss: 2.021132230758667
# Epoch [4/10], Loss: 2.1495
# Batch 个数 0, 当前Batch Loss: 1.8851388692855835
# Batch 个数 50, 当前Batch Loss: 1.9429242610931396
# Batch 个数 100, 当前Batch Loss: 1.7991572618484497
# Batch 个数 150, 当前Batch Loss: 1.9191457033157349
# Batch 个数 200, 当前Batch Loss: 1.9261804819107056
# Batch 个数 250, 当前Batch Loss: 1.8450063467025757
# Batch 个数 300, 当前Batch Loss: 1.904905080795288
# Batch 个数 350, 当前Batch Loss: 1.55928635597229
# Epoch [5/10], Loss: 1.8360
# Batch 个数 0, 当前Batch Loss: 1.3336247205734253
# Batch 个数 50, 当前Batch Loss: 1.3362265825271606
# Batch 个数 100, 当前Batch Loss: 1.709336757659912
# Batch 个数 150, 当前Batch Loss: 1.2398942708969116
# Batch 个数 200, 当前Batch Loss: 1.892342209815979
# Batch 个数 250, 当前Batch Loss: 1.5623533725738525
# Batch 个数 300, 当前Batch Loss: 1.4189115762710571
# Batch 个数 350, 当前Batch Loss: 1.325492024421692
# Epoch [6/10], Loss: 1.4336
# Batch 个数 0, 当前Batch Loss: 1.2600481510162354
# Batch 个数 50, 当前Batch Loss: 1.0597386360168457
# Batch 个数 100, 当前Batch Loss: 1.06895112991333
# Batch 个数 150, 当前Batch Loss: 1.0308781862258911
# Batch 个数 200, 当前Batch Loss: 1.122162938117981
# Batch 个数 250, 当前Batch Loss: 1.3941481113433838
# Batch 个数 300, 当前Batch Loss: 0.9339541792869568
# Batch 个数 350, 当前Batch Loss: 1.0344043970108032
# Epoch [7/10], Loss: 1.0896
# Batch 个数 0, 当前Batch Loss: 0.8986969590187073
# Batch 个数 50, 当前Batch Loss: 0.9678691029548645
# Batch 个数 100, 当前Batch Loss: 1.058298945426941
# Batch 个数 150, 当前Batch Loss: 0.9985963702201843
# Batch 个数 200, 当前Batch Loss: 0.6239901185035706
# Batch 个数 250, 当前Batch Loss: 0.6054133176803589
# Batch 个数 300, 当前Batch Loss: 0.6963202953338623
# Batch 个数 350, 当前Batch Loss: 0.7411804795265198
# Epoch [8/10], Loss: 0.8451
# Batch 个数 0, 当前Batch Loss: 0.760347306728363
# Batch 个数 50, 当前Batch Loss: 0.7765561938285828
# Batch 个数 100, 当前Batch Loss: 0.7310798168182373
# Batch 个数 150, 当前Batch Loss: 0.7559653520584106
# Batch 个数 200, 当前Batch Loss: 0.4455700218677521
# Batch 个数 250, 当前Batch Loss: 0.7218654751777649
# Batch 个数 300, 当前Batch Loss: 0.6985408067703247
# Batch 个数 350, 当前Batch Loss: 0.7245393395423889
# Epoch [9/10], Loss: 0.6897
# Batch 个数 0, 当前Batch Loss: 0.8590896129608154
# Batch 个数 50, 当前Batch Loss: 0.3893505334854126
# Batch 个数 100, 当前Batch Loss: 0.4267650842666626
# Batch 个数 150, 当前Batch Loss: 0.6829278469085693
# Batch 个数 200, 当前Batch Loss: 0.7947127819061279
# Batch 个数 250, 当前Batch Loss: 0.6016663908958435
# Batch 个数 300, 当前Batch Loss: 0.6200481653213501
# Batch 个数 350, 当前Batch Loss: 0.5195834636688232
# Epoch [10/10], Loss: 0.5930
# 输入 '帮我导航到北京' 预测为: 'Travel-Query'
# 输入 '查询明天北京的天气' 预测为: 'Weather-Query'
