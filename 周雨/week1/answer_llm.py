
from openai import OpenAI



client = OpenAI(
    api_key="sk-8e9ca82a51a24bfd924f7969c53786f3", # 账号绑定，用来计费的

    # 大模型厂商的地址，阿里云
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)



completion = client.chat.completions.create(
    model="qwen-flash",
                                            messages=[
            {"role": "user", "content": f"""帮我进行文本分类：带我去郭德纲家看喜羊羊与灰太狼

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
"""}])

print(completion.choices[0].message.content)