# depth-estimation

First, input the single image into Generator and predict the initial surface normal and initial depth in the Generator.
However, by mutually constraining these two preliminary results, we generate a single optimized depth map. This depth map, along with the surface normal map, is then input into the discriminator. Additionally, the ground truth is also provided to the discriminator as reference data for evaluation.
![流程](https://github.com/RBing123/depth-estimation/assets/107789113/29c6a7be-8992-4834-b316-6cee2329cb27)
# Depth to Normal
![新增 Microsoft PowerPoint 簡報](https://github.com/RBing123/depth-estimation/assets/107789113/bace7707-d261-4413-bea9-622dab2b6d79)
# Normal to depth
The principle involves assigning each pixel an initial normal vector and an initial estimated depth, which can satisfy the plane equation. I need to assume that the given initial depth map is accurate and that the neighboring pixels also lie on this plane. Two hyperparameters can be used to define what constitutes neighboring pixels. Then, using the perspective projection formula, the depth is recalculated for all pixels and their corresponding depths, and all pixels are aggregated using kernel regression.

# Architecture
![image](https://github.com/RBing123/depth-estimation/assets/107789113/5bd3f168-0868-42de-a5f8-c2cb67f9d38e)
# Result
![image](https://github.com/RBing123/depth-estimation/assets/107789113/c812e7ab-f60c-4785-9ba8-e7a20c2ef8af)
![image](https://github.com/RBing123/depth-estimation/assets/107789113/c6d8c446-1db0-41ac-a23d-cec24c1e07ed)
![image](https://github.com/RBing123/depth-estimation/assets/107789113/adba2f4f-18e7-403d-b408-15b73321b4c8)
![image](https://github.com/RBing123/depth-estimation/assets/107789113/82b73d0c-ad83-4c67-8d84-1eab0f57f7b1)
![image](https://github.com/RBing123/depth-estimation/assets/107789113/9eab39a0-e661-494e-bbde-757fe578c11f)


Far---------------------------------------------------------------------------------------------->Near

The above picture from left to right shows a single RGB image, ground truth, and model-estimated depth map.
![image](https://github.com/RBing123/depth-estimation/assets/107789113/0ffe3af7-7ed2-422b-8753-f898ee96ce69)
![image](https://github.com/RBing123/depth-estimation/assets/107789113/7d9f237a-4b7e-40ac-8fc5-3aeeca1c1e31)
![image](https://github.com/RBing123/depth-estimation/assets/107789113/153e88a2-7627-45ad-a8d8-7b69f18e5467)
![image](https://github.com/RBing123/depth-estimation/assets/107789113/c97f0529-13b2-410f-9a45-041f7f0ac3e9)

The above picture from left to right shows the surface normal vector map of a single RGB image, ground truth, and model estimation.

# Real Test
https://drive.google.com/file/d/1BIzOxXL_kvY8E0uERxxWKg-qQUWaIa8_/view?usp=drive_link






