import pandas as pd
import jieba # 中文分词器
from sklearn.feature_extraction.text import CountVectorizer # 词频统计
from sklearn.neighbors import KNeighborsClassifier # KNN
from openai import OpenAI

API="sk-848682a3757d4e21bab0d21217ffbb26"
Baseurl="https://dashscope.aliyuncs.com/compatible-mode/v1"

class TextClassfy:
    """
    Docstring for TextClassfy
    文本分类器
    """
    def __init__(self,filename:str,api_key:str,base_url):
        """
        Docstring for __init__
        
        :param self: Description
        :param filename: 数据集名称
        :type filename: str
        :param api_key: Description
        :type api_key: str
        """
        self.datafile=filename
        # 机器学习模型
        self.dataset = pd.read_csv(self.datafile, sep="\t", names=["text","label"], header=None, nrows=None)
        self.knn_model=KNeighborsClassifier()
        # 大模型API
        self.client = OpenAI(
            # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
            # https://bailian.console.aliyun.com/?tab=model#/api-key
            api_key=api_key, # 账号绑定，用来计费的

            # 大模型厂商的地址，阿里云
            base_url=base_url,
        )
        self.vector=CountVectorizer() #词频统计器

    def feature_extrator(self):
        input_sententce = self.dataset["text"].apply(lambda x: " ".join(jieba.lcut(x))) # sklearn对中文处理
        self.vector.fit(input_sententce.values) # 统计词表
        input_feature=self.vector.transform(input_sententce.values)
        return input_feature
    
    def text_calssify_using_ml(self, text: str) -> str:
        self.knn_model.fit(self.feature_extrator(), self.dataset["label"].values)
        test_sentence = " ".join(jieba.lcut(text))
        test_feature = self.vector.transform([test_sentence])
        return self.knn_model.predict(test_feature)[0]
    
    def text_calssify_using_llm(self, text: str) -> str:
        """
        文本分类（大语言模型），输入文本完成类别划分
        """
        completion = self.client.chat.completions.create(
            model="qwen-flash",  # 模型的代号
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
                """},  # 用户的提问
                        ]
        )
        return completion.choices[0].message.content
    
if __name__ == "__main__":
    textclassfier=TextClassfy("dataset.csv",API,Baseurl)
    ml_result=textclassfier.text_calssify_using_ml("帮我导航到磨子桥")
    llm_result=textclassfier.text_calssify_using_llm("帮我导航到磨子桥")
    print(f"机器识别结果{ml_result}")
    print(f"大模型识别结果{llm_result}")
