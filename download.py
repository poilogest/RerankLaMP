import os
import requests
# 下载lamp数据集
tasks = [1, 2, 3, 4, 5, 6, 7]
splits = ["train", "dev"]

for task in tasks:
    os.makedirs(f"LaMP_{task}", exist_ok=True)
    os.chdir(f"LaMP_{task}")

    for split in splits:
        for file in ["questions", "outputs"]:
            url = f"https://ciir.cs.umass.edu/downloads/LaMP/LaMP_{task}/{split}/{split}_{file}.json"
            response = requests.get(url)
            with open(f"{split}_{file}.json", "wb") as f:
                f.write(response.content)

    url = f"https://ciir.cs.umass.edu/downloads/LaMP/LaMP_{task}/test/test_questions.json"
    response = requests.get(url)
    with open("test_questions.json", "wb") as f:
        f.write(response.content)

    os.chdir("..")