import pandas as pd
import jieba
from sklearn.feature_extraction.text import CountVectorizer  # 词频统计
from sklearn.neighbors import KNeighborsClassifier  # KNN
from openai import OpenAI

dataset = pd.read_csv("dataset.csv", sep="\t", header=None, nrows=10000)
print(dataset[1].value_counts())

input_sententce = dataset[0].apply(lambda x: " ".join(jieba.lcut(x)))  # sklearn对中文处理

vector = CountVectorizer()  # 对文本进行提取特征 默认是使用标点符号分词， 不是模型
vector.fit(input_sententce.values)  # 统计词表
input_feature = vector.transform(input_sententce.values)  # 进行转换 100 * 词表大小

model = KNeighborsClassifier()
model.fit(input_feature, dataset[1].values)
client = OpenAI(
    api_key="sk-4cdef86c739746f9b6bac72ecfd213f6",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


def text_calssify_using_llm(text: str) -> str:
    completion = client.chat.completions.create(
        model="qwen-flash",

        messages=[
            {"role": "user", "content": f"""帮我进行文本分类：{text}
    输出的类别只能从如下中进行选择， 除了类别之外下列的类别，请给出最合适的类别。
FilmTele-Play            
Video-Play               
Music-Play              
Radio-Listen           
Alarm-Update        
Travel-Query        
HomeAppliance-Control  
Weather-Query          
Calendar-Query      
TVProgram-Play      
Audio-Play       
Other             
"""},
        ]
    )
    return completion.choices[0].message.content


if __name__ == "__main__":
 print("2222")
 print("大语言模型: ", text_calssify_using_llm("帮我导航到天安门"))
