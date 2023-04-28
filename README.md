# Masterthesis_SLM-CNN
Masterthesis on the topic of error/anomaly detection in 3D printing of metal structures. 

The Proceedings Paper and CNN only part can be found here: 

https://github.com/Ay-De/SLM-CNN


The main difference between the proceedings paper and the complete master thesis is the real time prediction of the entire powderbed. This prediction takes not only the CNN output into account, but the prediction results of the previous printed layers. Rationale behind this approach is, that errors in the printing process are developing over time.

Example of classifications during printing of an object.

Note: Green patches are classified as powder, Blue patches are classified as printed objects, Red patches are classified to have an error.

![Example 1](https://user-images.githubusercontent.com/91670696/235149734-b51c9ba8-006e-44b7-b5ad-220a0be98637.mp4)

![Example 2](https://user-images.githubusercontent.com/91670696/235150196-abaef681-b708-486c-a4bc-650b468b870f.mp4)
