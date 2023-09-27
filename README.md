# depth-estimation

將單張RGB影像輸入到生成器中，而在生成器中會分別進行表面法向量估計與深度估計的初步估計，之後將這兩張初步成果進行互相約制產製單張優化後的深度圖，將這張深度圖與表面法向量圖輸入到判別器中，同時將地面真值也輸入到判別器中做為判斷的資料
![流程](https://github.com/RBing123/depth-estimation/assets/107789113/29c6a7be-8992-4834-b316-6cee2329cb27)
深度到法向量的轉換
![新增 Microsoft PowerPoint 簡報](https://github.com/RBing123/depth-estimation/assets/107789113/bace7707-d261-4413-bea9-622dab2b6d79)


