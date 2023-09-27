# depth-estimation

將單張RGB影像輸入到生成器中，而在生成器中會分別進行表面法向量估計與深度估計的初步估計，之後將這兩張初步成果進行互相約制產製單張優化後的深度圖，將這張深度圖與表面法向量圖輸入到判別器中，同時將地面真值也輸入到判別器中做為判斷的資料
![流程](https://github.com/RBing123/depth-estimation/assets/107789113/29c6a7be-8992-4834-b316-6cee2329cb27)
# 深度到法向量的轉換
![新增 Microsoft PowerPoint 簡報](https://github.com/RBing123/depth-estimation/assets/107789113/bace7707-d261-4413-bea9-622dab2b6d79)
# 法向量到深度的轉換
原理為針對每一個像素給定每一個初始像素的法向量以及初始估計的深度，並且可以滿足切平
面方程式，而我需要假設給定的初始深度圖是準確的並且假設這個像素的鄰近也都落在這個
切平面上，可以透過兩個超參數來決定何謂鄰近的像素，然後將所有的像素與像素對應的深
度，利用透視投影的公式反算深度，並且利用核迴歸將所有的像素聚合。

# 神經網路架構
![image](https://github.com/RBing123/depth-estimation/assets/107789113/5bd3f168-0868-42de-a5f8-c2cb67f9d38e)
# 訓練成果
![image](https://github.com/RBing123/depth-estimation/assets/107789113/c812e7ab-f60c-4785-9ba8-e7a20c2ef8af)
![image](https://github.com/RBing123/depth-estimation/assets/107789113/c6d8c446-1db0-41ac-a23d-cec24c1e07ed)
![image](https://github.com/RBing123/depth-estimation/assets/107789113/adba2f4f-18e7-403d-b408-15b73321b4c8)
![image](https://github.com/RBing123/depth-estimation/assets/107789113/82b73d0c-ad83-4c67-8d84-1eab0f57f7b1)
上圖從左到右分別為單張RGB影像、地面真值、模型估計的深度圖
![image](https://github.com/RBing123/depth-estimation/assets/107789113/0ffe3af7-7ed2-422b-8753-f898ee96ce69)
![image](https://github.com/RBing123/depth-estimation/assets/107789113/7d9f237a-4b7e-40ac-8fc5-3aeeca1c1e31)
![image](https://github.com/RBing123/depth-estimation/assets/107789113/153e88a2-7627-45ad-a8d8-7b69f18e5467)
![image](https://github.com/RBing123/depth-estimation/assets/107789113/c97f0529-13b2-410f-9a45-041f7f0ac3e9)
上圖從左到右分別為單張RGB影像、地面真值、模型估計的表面法向量圖

# 實際成果
<source src="https://drive.google.com/file/d/1BIzOxXL_kvY8E0uERxxWKg-qQUWaIa8_/view?usp=drive_link" type="video/mp4">






