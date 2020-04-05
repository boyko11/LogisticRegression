Logistic Regression for the Wisconsin Breast Cancer Dataset:

https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)

## Training Learning Curve

![Training Learning Curve](training_learning_curve.png)

## Accuracy Stats

Overall Accuracy: 0.928

| Class         | Labeled       | Predicted|
|:-------------:|:-------------:|:-------------:|
| Malignant     | 212           | 191           |
| Benign        | 357           | 378           |

| Class         | True Positive | False Positive|
|:-------------:|:-------------:|:-------------:|
| Malignant     | 183           | 8             |
| Benign        | 349           | 29            |

| Class         | Precision     | Recall| F1 Score |
|:-------------:|:-------------:|:-----:|:--------:|
| Malignant     | 0.958         | 0.863 | 0.908    |
| Benign        | 0.923         | 0.978 | 0.950    |

|Record ID|Label| Predicted Malignant Probability| Absolute Error| LogIt Error | Rounded Prediction Error |
|:-------:|:---:|:--------------------:|:-------------:|:-----------:|:----------------:|
|842302|Malignant|1.0|0.0|0.0|0.0|
|842517|Malignant|0.9719|0.0281|0.0285|0.0|
|84300903|Malignant|1.0|0.0|0.0|0.0|
|84348301|Malignant|1.0|0.0|0.0|0.0|
|84358402|Malignant|0.991|0.009|0.0091|0.0|
|843786|Malignant|0.9695|0.0305|0.031|0.0|
|844359|Malignant|0.988|0.012|0.0121|0.0|
|84458202|Malignant|0.9758|0.0242|0.0244|0.0|
|844981|Malignant|0.9988|0.0012|0.0012|0.0|
|84501001|Malignant|1.0|0.0|0.0|0.0|
|845636|Malignant|0.1586|0.8414|1.8416|1.0|
|84610002|Malignant|0.9819|0.0181|0.0183|0.0|
|846226|Malignant|1.0|0.0|0.0|0.0|
|846381|Malignant|0.2827|0.7173|1.2635|1.0|
|84667401|Malignant|0.9999|0.0001|0.0001|0.0|
|84799002|Malignant|1.0|0.0|0.0|0.0|
|848406|Malignant|0.5483|0.4517|0.6009|0.0|
|84862001|Malignant|0.9999|0.0001|0.0001|0.0|
|849014|Malignant|1.0|0.0|0.0|0.0|
|8510426|Benign|0.0043|0.0043|-0.0043|0.0|
|8510653|Benign|0.0018|0.0018|-0.0018|0.0|
|8510824|Benign|0.0|0.0|0.0|0.0|
|8511133|Malignant|0.9976|0.0024|0.0024|0.0|
|851509|Malignant|1.0|0.0|0.0|0.0|
|852552|Malignant|1.0|0.0|0.0|0.0|
|852631|Malignant|1.0|0.0|0.0|0.0|
|852763|Malignant|0.9996|0.0004|0.0004|0.0|
|852781|Malignant|0.9908|0.0092|0.0092|0.0|
|852973|Malignant|1.0|0.0|0.0|0.0|
|853201|Malignant|0.4115|0.5885|0.8881|1.0|
|853401|Malignant|1.0|0.0|0.0|0.0|
|853612|Malignant|0.9937|0.0063|0.0063|0.0|
|85382601|Malignant|1.0|0.0|0.0|0.0|
|854002|Malignant|1.0|0.0|0.0|0.0|
|854039|Malignant|0.9963|0.0037|0.0037|0.0|
|854253|Malignant|0.998|0.002|0.002|0.0|
|854268|Malignant|0.8967|0.1033|0.1091|0.0|
|854941|Benign|0.0|0.0|0.0|0.0|
|855133|Malignant|0.0071|0.9929|4.9447|1.0|
|855138|Malignant|0.7196|0.2804|0.3291|0.0|
|855167|Malignant|0.0029|0.9971|5.8479|1.0|
|855563|Malignant|0.7185|0.2815|0.3305|0.0|
|855625|Malignant|1.0|0.0|0.0|0.0|
|856106|Malignant|0.798|0.202|0.2257|0.0|
|85638502|Malignant|0.3858|0.6142|0.9525|1.0|
|857010|Malignant|1.0|0.0|0.0|0.0|
|85713702|Benign|0.0|0.0|0.0|0.0|
|85715|Malignant|0.9694|0.0306|0.0311|0.0|
|857155|Benign|0.0008|0.0008|-0.0008|0.0|
|857156|Benign|0.0163|0.0163|-0.0164|0.0|
|857343|Benign|0.0|0.0|0.0|0.0|
|857373|Benign|0.0001|0.0001|-0.0001|0.0|
|857374|Benign|0.0|0.0|0.0|0.0|
|857392|Malignant|0.9856|0.0144|0.0145|0.0|
|857438|Malignant|0.1912|0.8088|1.6543|1.0|
|85759902|Benign|0.0002|0.0002|-0.0002|0.0|
|857637|Malignant|1.0|0.0|0.0|0.0|
|857793|Malignant|0.9823|0.0177|0.0178|0.0|
|857810|Benign|0.0|0.0|0.0|0.0|
|858477|Benign|0.0|0.0|0.0|0.0|
|858970|Benign|0.0001|0.0001|-0.0001|0.0|
|858981|Benign|0.0008|0.0008|-0.0008|0.0|
|858986|Malignant|0.9996|0.0004|0.0004|0.0|
|859196|Benign|0.0|0.0|0.0|0.0|
|85922302|Malignant|0.9917|0.0083|0.0083|0.0|
|859283|Malignant|0.9911|0.0089|0.0089|0.0|
|859464|Benign|0.0013|0.0013|-0.0013|0.0|
|859465|Benign|0.0|0.0|0.0|0.0|
|859471|Benign|0.9679|0.9679|-3.4378|1.0|
|859487|Benign|0.0001|0.0001|-0.0001|0.0|
|859575|Malignant|0.9862|0.0138|0.0139|0.0|
|859711|Benign|0.0001|0.0001|-0.0001|0.0|
|859717|Malignant|1.0|0.0|0.0|0.0|
|859983|Malignant|0.031|0.969|3.4745|1.0|
|8610175|Benign|0.0002|0.0002|-0.0002|0.0|
|8610404|Malignant|0.7995|0.2005|0.2238|0.0|
|8610629|Benign|0.001|0.001|-0.001|0.0|
|8610637|Malignant|0.9999|0.0001|0.0001|0.0|
|8610862|Malignant|1.0|0.0|0.0|0.0|
|8610908|Benign|0.0016|0.0016|-0.0016|0.0|
|861103|Benign|0.0188|0.0188|-0.019|0.0|
|8611161|Benign|0.5132|0.5132|-0.7199|1.0|
|8611555|Malignant|1.0|0.0|0.0|0.0|
|8611792|Malignant|1.0|0.0|0.0|0.0|
|8612080|Benign|0.001|0.001|-0.001|0.0|
|8612399|Malignant|0.9975|0.0025|0.0025|0.0|
|86135501|Malignant|0.5171|0.4829|0.6596|0.0|
|86135502|Malignant|0.9997|0.0003|0.0003|0.0|
|861597|Benign|0.0342|0.0342|-0.0348|0.0|
|861598|Benign|0.1416|0.1416|-0.1527|0.0|
|861648|Benign|0.0094|0.0094|-0.0094|0.0|
|861799|Malignant|0.2206|0.7794|1.5114|1.0|
|861853|Benign|0.0001|0.0001|-0.0001|0.0|
|862009|Benign|0.0031|0.0031|-0.0031|0.0|
|862028|Malignant|0.9842|0.0158|0.0159|0.0|
|86208|Malignant|0.9998|0.0002|0.0002|0.0|
|86211|Benign|0.0002|0.0002|-0.0002|0.0|
|862261|Benign|0.0|0.0|0.0|0.0|
|862485|Benign|0.0|0.0|0.0|0.0|
|862548|Malignant|0.5993|0.4007|0.5119|0.0|
|862717|Malignant|0.2898|0.7102|1.2387|1.0|
|862722|Benign|0.0|0.0|0.0|0.0|
|862965|Benign|0.0002|0.0002|-0.0002|0.0|
|862980|Benign|0.0024|0.0024|-0.0024|0.0|
|862989|Benign|0.0001|0.0001|-0.0001|0.0|
|863030|Malignant|0.9957|0.0043|0.0043|0.0|
|863031|Benign|0.0997|0.0997|-0.105|0.0|
|863270|Benign|0.0002|0.0002|-0.0002|0.0|
|86355|Malignant|1.0|0.0|0.0|0.0|
|864018|Benign|0.0051|0.0051|-0.0051|0.0|
|864033|Benign|0.0001|0.0001|-0.0001|0.0|
|86408|Benign|0.0494|0.0494|-0.0507|0.0|
|86409|Benign|0.8375|0.8375|-1.817|1.0|
|864292|Benign|0.002|0.002|-0.002|0.0|
|864496|Benign|0.0003|0.0003|-0.0003|0.0|
|864685|Benign|0.0032|0.0032|-0.0032|0.0|
|864726|Benign|0.0|0.0|0.0|0.0|
|864729|Malignant|0.998|0.002|0.002|0.0|
|864877|Malignant|1.0|0.0|0.0|0.0|
|865128|Malignant|0.5658|0.4342|0.5696|0.0|
|865137|Benign|0.0|0.0|0.0|0.0|
|86517|Malignant|0.9962|0.0038|0.0038|0.0|
|865423|Malignant|1.0|0.0|0.0|0.0|
|865432|Benign|0.0035|0.0035|-0.0035|0.0|
|865468|Benign|0.0002|0.0002|-0.0002|0.0|
|86561|Benign|0.0001|0.0001|-0.0001|0.0|
|866083|Malignant|0.4392|0.5608|0.8228|1.0|
|866203|Malignant|0.7961|0.2039|0.228|0.0|
|866458|Benign|0.4185|0.4185|-0.5422|0.0|
|866674|Malignant|1.0|0.0|0.0|0.0|
|866714|Benign|0.0005|0.0005|-0.0005|0.0|
|8670|Malignant|0.9091|0.0909|0.0953|0.0|
|86730502|Malignant|0.9448|0.0552|0.0568|0.0|
|867387|Benign|0.0083|0.0083|-0.0083|0.0|
|867739|Malignant|0.9925|0.0075|0.0075|0.0|
|868202|Malignant|0.0149|0.9851|4.2091|1.0|
|868223|Benign|0.0003|0.0003|-0.0003|0.0|
|868682|Benign|0.0|0.0|0.0|0.0|
|868826|Malignant|0.9728|0.0272|0.0275|0.0|
|868871|Benign|0.0003|0.0003|-0.0003|0.0|
|868999|Benign|0.0|0.0|0.0|0.0|
|869104|Malignant|0.617|0.383|0.4828|0.0|
|869218|Benign|0.0009|0.0009|-0.0009|0.0|
|869224|Benign|0.0012|0.0012|-0.0012|0.0|
|869254|Benign|0.0|0.0|0.0|0.0|
|869476|Benign|0.0011|0.0011|-0.0011|0.0|
|869691|Malignant|0.9213|0.0787|0.0819|0.0|
|86973701|Benign|0.0417|0.0417|-0.0426|0.0|
|86973702|Benign|0.0185|0.0185|-0.0186|0.0|
|869931|Benign|0.0001|0.0001|-0.0001|0.0|
|871001501|Benign|0.0073|0.0073|-0.0074|0.0|
|871001502|Benign|0.1093|0.1093|-0.1157|0.0|
|8710441|Benign|0.9992|0.9992|-7.0847|1.0|
|87106|Benign|0.0|0.0|0.0|0.0|
|8711002|Benign|0.0091|0.0091|-0.0092|0.0|
|8711003|Benign|0.0004|0.0004|-0.0004|0.0|
|8711202|Malignant|0.9981|0.0019|0.0019|0.0|
|8711216|Benign|0.0114|0.0114|-0.0115|0.0|
|871122|Benign|0.0|0.0|0.0|0.0|
|871149|Benign|0.0|0.0|0.0|0.0|
|8711561|Benign|0.0218|0.0218|-0.0221|0.0|
|8711803|Malignant|0.9375|0.0625|0.0646|0.0|
|871201|Malignant|1.0|0.0|0.0|0.0|
|8712064|Benign|0.0153|0.0153|-0.0154|0.0|
|8712289|Malignant|1.0|0.0|0.0|0.0|
|8712291|Benign|0.0003|0.0003|-0.0003|0.0|
|87127|Benign|0.0|0.0|0.0|0.0|
|8712729|Malignant|0.5894|0.4106|0.5287|0.0|
|8712766|Malignant|0.9999|0.0001|0.0001|0.0|
|8712853|Benign|0.0015|0.0015|-0.0015|0.0|
|87139402|Benign|0.0001|0.0001|-0.0001|0.0|
|87163|Malignant|0.0511|0.9489|2.9746|1.0|
|87164|Malignant|0.9378|0.0622|0.0642|0.0|
|871641|Benign|0.0|0.0|0.0|0.0|
|871642|Benign|0.0|0.0|0.0|0.0|
|872113|Benign|0.0|0.0|0.0|0.0|
|872608|Benign|0.0417|0.0417|-0.0426|0.0|
|87281702|Malignant|0.9939|0.0061|0.0061|0.0|
|873357|Benign|0.0|0.0|0.0|0.0|
|873586|Benign|0.0|0.0|0.0|0.0|
|873592|Malignant|1.0|0.0|0.0|0.0|
|873593|Malignant|1.0|0.0|0.0|0.0|
|873701|Malignant|0.8572|0.1428|0.1541|0.0|
|873843|Benign|0.0|0.0|0.0|0.0|
|873885|Malignant|0.2163|0.7837|1.5311|1.0|
|874158|Benign|0.0|0.0|0.0|0.0|
|874217|Malignant|0.447|0.553|0.8052|1.0|
|874373|Benign|0.0002|0.0002|-0.0002|0.0|
|874662|Benign|0.0002|0.0002|-0.0002|0.0|
|874839|Benign|0.0|0.0|0.0|0.0|
|874858|Malignant|1.0|0.0|0.0|0.0|
|875093|Benign|0.0005|0.0005|-0.0005|0.0|
|875099|Benign|0.0|0.0|0.0|0.0|
|875263|Malignant|0.9961|0.0039|0.0039|0.0|
|87556202|Malignant|0.9707|0.0293|0.0297|0.0|
|875878|Benign|0.0001|0.0001|-0.0001|0.0|
|875938|Malignant|0.995|0.005|0.005|0.0|
|877159|Malignant|0.2096|0.7904|1.5627|1.0|
|877486|Malignant|0.9994|0.0006|0.0006|0.0|
|877500|Malignant|0.9571|0.0429|0.0439|0.0|
|877501|Benign|0.0113|0.0113|-0.0114|0.0|
|877989|Malignant|0.8981|0.1019|0.1074|0.0|
|878796|Malignant|1.0|0.0|0.0|0.0|
|87880|Malignant|1.0|0.0|0.0|0.0|
|87930|Benign|0.0248|0.0248|-0.0251|0.0|
|879523|Malignant|0.0865|0.9135|2.4472|1.0|
|879804|Benign|0.0001|0.0001|-0.0001|0.0|
|879830|Malignant|0.2492|0.7508|1.3894|1.0|
|8810158|Benign|0.4065|0.4065|-0.5217|0.0|
|8810436|Benign|0.0003|0.0003|-0.0003|0.0|
|881046502|Malignant|0.9998|0.0002|0.0002|0.0|
|8810528|Benign|0.0002|0.0002|-0.0002|0.0|
|8810703|Malignant|1.0|0.0|0.0|0.0|
|881094802|Malignant|0.9924|0.0076|0.0076|0.0|
|8810955|Malignant|0.9955|0.0045|0.0046|0.0|
|8810987|Malignant|0.7305|0.2695|0.314|0.0|
|8811523|Benign|0.0382|0.0382|-0.0389|0.0|
|8811779|Benign|0.0|0.0|0.0|0.0|
|8811842|Malignant|0.9999|0.0001|0.0001|0.0|
|88119002|Malignant|1.0|0.0|0.0|0.0|
|8812816|Benign|0.0001|0.0001|-0.0001|0.0|
|8812818|Benign|0.0032|0.0032|-0.0032|0.0|
|8812844|Benign|0.0001|0.0001|-0.0001|0.0|
|8812877|Malignant|0.9812|0.0188|0.019|0.0|
|8813129|Benign|0.0004|0.0004|-0.0004|0.0|
|88143502|Benign|0.0035|0.0035|-0.0035|0.0|
|88147101|Benign|0.0|0.0|0.0|0.0|
|88147102|Benign|0.0082|0.0082|-0.0082|0.0|
|88147202|Benign|0.0198|0.0198|-0.02|0.0|
|881861|Malignant|0.9971|0.0029|0.0029|0.0|
|881972|Malignant|0.9986|0.0014|0.0014|0.0|
|88199202|Benign|0.0001|0.0001|-0.0001|0.0|
|88203002|Benign|0.0017|0.0017|-0.0017|0.0|
|88206102|Malignant|0.9998|0.0002|0.0002|0.0|
|882488|Benign|0.0|0.0|0.0|0.0|
|88249602|Benign|0.0036|0.0036|-0.0036|0.0|
|88299702|Malignant|1.0|0.0|0.0|0.0|
|883263|Malignant|0.9839|0.0161|0.0162|0.0|
|883270|Benign|0.223|0.223|-0.2524|0.0|
|88330202|Malignant|1.0|0.0|0.0|0.0|
|88350402|Benign|0.0003|0.0003|-0.0003|0.0|
|883539|Benign|0.0|0.0|0.0|0.0|
|883852|Benign|0.3608|0.3608|-0.4475|0.0|
|88411702|Benign|0.0012|0.0012|-0.0012|0.0|
|884180|Malignant|0.9997|0.0003|0.0003|0.0|
|884437|Benign|0.0024|0.0024|-0.0024|0.0|
|884448|Benign|0.0001|0.0001|-0.0001|0.0|
|884626|Benign|0.0604|0.0604|-0.0623|0.0|
|88466802|Benign|0.0082|0.0082|-0.0082|0.0|
|884689|Benign|0.0003|0.0003|-0.0003|0.0|
|884948|Malignant|1.0|0.0|0.0|0.0|
|88518501|Benign|0.0001|0.0001|-0.0001|0.0|
|885429|Malignant|1.0|0.0|0.0|0.0|
|8860702|Malignant|0.8493|0.1507|0.1633|0.0|
|886226|Malignant|0.9995|0.0005|0.0005|0.0|
|886452|Malignant|0.232|0.768|1.4611|1.0|
|88649001|Malignant|1.0|0.0|0.0|0.0|
|886776|Malignant|0.9999|0.0001|0.0001|0.0|
|887181|Malignant|1.0|0.0|0.0|0.0|
|88725602|Malignant|1.0|0.0|0.0|0.0|
|887549|Malignant|1.0|0.0|0.0|0.0|
|888264|Malignant|0.1383|0.8617|1.9786|1.0|
|888570|Malignant|0.982|0.018|0.0182|0.0|
|889403|Malignant|0.0053|0.9947|5.2326|1.0|
|889719|Malignant|0.9812|0.0188|0.0189|0.0|
|88995002|Malignant|1.0|0.0|0.0|0.0|
|8910251|Benign|0.0007|0.0007|-0.0007|0.0|
|8910499|Benign|0.0014|0.0014|-0.0015|0.0|
|8910506|Benign|0.0005|0.0005|-0.0005|0.0|
|8910720|Benign|0.0093|0.0093|-0.0093|0.0|
|8910721|Benign|0.0|0.0|0.0|0.0|
|8910748|Benign|0.0|0.0|0.0|0.0|
|8910988|Malignant|1.0|0.0|0.0|0.0|
|8910996|Benign|0.0|0.0|0.0|0.0|
|8911163|Malignant|0.8205|0.1795|0.1979|0.0|
|8911164|Benign|0.0032|0.0032|-0.0032|0.0|
|8911230|Benign|0.0|0.0|0.0|0.0|
|8911670|Malignant|0.2537|0.7463|1.3718|1.0|
|8911800|Benign|0.0|0.0|0.0|0.0|
|8911834|Benign|0.0008|0.0008|-0.0008|0.0|
|8912049|Malignant|1.0|0.0|0.0|0.0|
|8912055|Benign|0.0|0.0|0.0|0.0|
|89122|Malignant|0.9996|0.0004|0.0004|0.0|
|8912280|Malignant|0.9752|0.0248|0.0251|0.0|
|8912284|Benign|0.0003|0.0003|-0.0003|0.0|
|8912521|Benign|0.0|0.0|0.0|0.0|
|8912909|Benign|0.0069|0.0069|-0.007|0.0|
|8913|Benign|0.0|0.0|0.0|0.0|
|8913049|Benign|0.0125|0.0125|-0.0125|0.0|
|89143601|Benign|0.0001|0.0001|-0.0001|0.0|
|89143602|Benign|0.4817|0.4817|-0.6572|0.0|
|8915|Benign|0.0711|0.0711|-0.0737|0.0|
|891670|Benign|0.0041|0.0041|-0.0042|0.0|
|891703|Benign|0.0003|0.0003|-0.0003|0.0|
|891716|Benign|0.0|0.0|0.0|0.0|
|891923|Benign|0.0|0.0|0.0|0.0|
|891936|Benign|0.0|0.0|0.0|0.0|
|892189|Malignant|0.0002|0.9998|8.3294|1.0|
|892214|Benign|0.0001|0.0001|-0.0001|0.0|
|892399|Benign|0.0|0.0|0.0|0.0|
|892438|Malignant|1.0|0.0|0.0|0.0|
|892604|Benign|0.0009|0.0009|-0.0009|0.0|
|89263202|Malignant|1.0|0.0|0.0|0.0|
|892657|Benign|0.0001|0.0001|-0.0001|0.0|
|89296|Benign|0.0|0.0|0.0|0.0|
|893061|Benign|0.0002|0.0002|-0.0002|0.0|
|89344|Benign|0.0|0.0|0.0|0.0|
|89346|Benign|0.0|0.0|0.0|0.0|
|893526|Benign|0.0|0.0|0.0|0.0|
|893548|Benign|0.0|0.0|0.0|0.0|
|893783|Benign|0.0001|0.0001|-0.0001|0.0|
|89382601|Benign|0.0|0.0|0.0|0.0|
|89382602|Benign|0.0001|0.0001|-0.0001|0.0|
|893988|Benign|0.0|0.0|0.0|0.0|
|894047|Benign|0.0|0.0|0.0|0.0|
|894089|Benign|0.0|0.0|0.0|0.0|
|894090|Benign|0.0|0.0|0.0|0.0|
|894326|Malignant|0.9599|0.0401|0.0409|0.0|
|894329|Benign|0.0878|0.0878|-0.0919|0.0|
|894335|Benign|0.0|0.0|0.0|0.0|
|894604|Benign|0.0011|0.0011|-0.0011|0.0|
|894618|Malignant|0.8386|0.1614|0.176|0.0|
|894855|Benign|0.0015|0.0015|-0.0015|0.0|
|895100|Malignant|1.0|0.0|0.0|0.0|
|89511501|Benign|0.0|0.0|0.0|0.0|
|89511502|Benign|0.0002|0.0002|-0.0002|0.0|
|89524|Benign|0.0|0.0|0.0|0.0|
|895299|Benign|0.0|0.0|0.0|0.0|
|8953902|Malignant|0.9925|0.0075|0.0075|0.0|
|895633|Malignant|0.907|0.093|0.0976|0.0|
|896839|Malignant|0.79|0.21|0.2357|0.0|
|896864|Benign|0.0157|0.0157|-0.0159|0.0|
|897132|Benign|0.0002|0.0002|-0.0002|0.0|
|897137|Benign|0.0|0.0|0.0|0.0|
|897374|Benign|0.0001|0.0001|-0.0001|0.0|
|89742801|Malignant|0.9986|0.0014|0.0014|0.0|
|897604|Benign|0.0001|0.0001|-0.0001|0.0|
|897630|Malignant|1.0|0.0|0.0|0.0|
|897880|Benign|0.0002|0.0002|-0.0002|0.0|
|89812|Malignant|1.0|0.0|0.0|0.0|
|89813|Benign|0.0707|0.0707|-0.0733|0.0|
|898143|Benign|0.0002|0.0002|-0.0002|0.0|
|89827|Benign|0.0004|0.0004|-0.0004|0.0|
|898431|Malignant|1.0|0.0|0.0|0.0|
|89864002|Benign|0.0004|0.0004|-0.0004|0.0|
|898677|Benign|0.0|0.0|0.0|0.0|
|898678|Benign|0.0001|0.0001|-0.0001|0.0|
|89869|Benign|0.0021|0.0021|-0.0021|0.0|
|898690|Benign|0.0001|0.0001|-0.0001|0.0|
|899147|Benign|0.0003|0.0003|-0.0003|0.0|
|899187|Benign|0.0|0.0|0.0|0.0|
|899667|Malignant|0.9999|0.0001|0.0001|0.0|
|899987|Malignant|1.0|0.0|0.0|0.0|
|9010018|Malignant|0.994|0.006|0.006|0.0|
|901011|Benign|0.0|0.0|0.0|0.0|
|9010258|Benign|0.0017|0.0017|-0.0017|0.0|
|9010259|Benign|0.1343|0.1343|-0.1442|0.0|
|901028|Benign|0.0001|0.0001|-0.0001|0.0|
|9010333|Benign|0.0|0.0|0.0|0.0|
|901034301|Benign|0.0|0.0|0.0|0.0|
|901034302|Benign|0.0|0.0|0.0|0.0|
|901041|Benign|0.0008|0.0008|-0.0008|0.0|
|9010598|Benign|0.0009|0.0009|-0.0009|0.0|
|9010872|Benign|0.041|0.041|-0.0419|0.0|
|9010877|Benign|0.0001|0.0001|-0.0001|0.0|
|901088|Malignant|0.9943|0.0057|0.0058|0.0|
|9011494|Malignant|1.0|0.0|0.0|0.0|
|9011495|Benign|0.0011|0.0011|-0.0011|0.0|
|9011971|Malignant|1.0|0.0|0.0|0.0|
|9012000|Malignant|1.0|0.0|0.0|0.0|
|9012315|Malignant|0.9998|0.0002|0.0002|0.0|
|9012568|Benign|0.0001|0.0001|-0.0001|0.0|
|9012795|Malignant|0.9932|0.0068|0.0068|0.0|
|901288|Malignant|0.9982|0.0018|0.0018|0.0|
|9013005|Benign|0.0001|0.0001|-0.0001|0.0|
|901303|Benign|0.0457|0.0457|-0.0467|0.0|
|901315|Benign|0.2794|0.2794|-0.3277|0.0|
|9013579|Benign|0.0021|0.0021|-0.0021|0.0|
|9013594|Benign|0.0012|0.0012|-0.0012|0.0|
|9013838|Malignant|0.9998|0.0002|0.0002|0.0|
|901549|Benign|0.0088|0.0088|-0.0089|0.0|
|901836|Benign|0.0|0.0|0.0|0.0|
|90250|Benign|0.0013|0.0013|-0.0013|0.0|
|90251|Benign|0.0277|0.0277|-0.0281|0.0|
|902727|Benign|0.0001|0.0001|-0.0001|0.0|
|90291|Malignant|0.1567|0.8433|1.8536|1.0|
|902975|Benign|0.0001|0.0001|-0.0001|0.0|
|902976|Benign|0.0|0.0|0.0|0.0|
|903011|Benign|0.0008|0.0008|-0.0008|0.0|
|90312|Malignant|0.9999|0.0001|0.0001|0.0|
|90317302|Benign|0.0|0.0|0.0|0.0|
|903483|Benign|0.0|0.0|0.0|0.0|
|903507|Malignant|0.9998|0.0002|0.0002|0.0|
|903516|Malignant|1.0|0.0|0.0|0.0|
|903554|Benign|0.0024|0.0024|-0.0024|0.0|
|903811|Benign|0.0002|0.0002|-0.0002|0.0|
|90401601|Benign|0.107|0.107|-0.1132|0.0|
|90401602|Benign|0.0001|0.0001|-0.0001|0.0|
|904302|Benign|0.0|0.0|0.0|0.0|
|904357|Benign|0.0002|0.0002|-0.0002|0.0|
|90439701|Malignant|1.0|0.0|0.0|0.0|
|904647|Benign|0.0|0.0|0.0|0.0|
|904689|Benign|0.0003|0.0003|-0.0003|0.0|
|9047|Benign|0.0004|0.0004|-0.0004|0.0|
|904969|Benign|0.0|0.0|0.0|0.0|
|904971|Benign|0.0004|0.0004|-0.0004|0.0|
|905189|Benign|0.0042|0.0042|-0.0042|0.0|
|905190|Benign|0.0014|0.0014|-0.0014|0.0|
|90524101|Malignant|0.9938|0.0062|0.0062|0.0|
|905501|Benign|0.0019|0.0019|-0.0019|0.0|
|905502|Benign|0.0012|0.0012|-0.0012|0.0|
|905520|Benign|0.0004|0.0004|-0.0004|0.0|
|905539|Benign|0.0|0.0|0.0|0.0|
|905557|Benign|0.2651|0.2651|-0.308|0.0|
|905680|Malignant|0.2228|0.7772|1.5015|1.0|
|905686|Benign|0.0027|0.0027|-0.0027|0.0|
|905978|Benign|0.0015|0.0015|-0.0015|0.0|
|90602302|Malignant|1.0|0.0|0.0|0.0|
|906024|Benign|0.0|0.0|0.0|0.0|
|906290|Benign|0.0002|0.0002|-0.0002|0.0|
|906539|Benign|0.0011|0.0011|-0.0011|0.0|
|906564|Benign|0.2277|0.2277|-0.2584|0.0|
|906616|Benign|0.0014|0.0014|-0.0014|0.0|
|906878|Benign|0.0335|0.0335|-0.0341|0.0|
|907145|Benign|0.0006|0.0006|-0.0006|0.0|
|907367|Benign|0.0|0.0|0.0|0.0|
|907409|Benign|0.0008|0.0008|-0.0008|0.0|
|90745|Benign|0.0022|0.0022|-0.0022|0.0|
|90769601|Benign|0.0|0.0|0.0|0.0|
|90769602|Benign|0.0|0.0|0.0|0.0|
|907914|Malignant|0.9998|0.0002|0.0002|0.0|
|907915|Benign|0.0083|0.0083|-0.0084|0.0|
|908194|Malignant|0.9999|0.0001|0.0001|0.0|
|908445|Malignant|0.9998|0.0002|0.0002|0.0|
|908469|Benign|0.0005|0.0005|-0.0005|0.0|
|908489|Malignant|0.8088|0.1912|0.2122|0.0|
|908916|Benign|0.0006|0.0006|-0.0006|0.0|
|909220|Benign|0.0004|0.0004|-0.0004|0.0|
|909231|Benign|0.0006|0.0006|-0.0006|0.0|
|909410|Benign|0.0|0.0|0.0|0.0|
|909411|Benign|0.0201|0.0201|-0.0203|0.0|
|909445|Malignant|0.9969|0.0031|0.0031|0.0|
|90944601|Benign|0.0|0.0|0.0|0.0|
|909777|Benign|0.0|0.0|0.0|0.0|
|9110127|Malignant|0.518|0.482|0.6578|0.0|
|9110720|Benign|0.032|0.032|-0.0325|0.0|
|9110732|Malignant|0.9999|0.0001|0.0001|0.0|
|9110944|Benign|0.0044|0.0044|-0.0044|0.0|
|911150|Benign|0.0093|0.0093|-0.0094|0.0|
|911157302|Malignant|0.9999|0.0001|0.0001|0.0|
|9111596|Benign|0.0016|0.0016|-0.0016|0.0|
|9111805|Malignant|0.998|0.002|0.002|0.0|
|9111843|Benign|0.0114|0.0114|-0.0114|0.0|
|911201|Benign|0.0028|0.0028|-0.0028|0.0|
|911202|Benign|0.0002|0.0002|-0.0002|0.0|
|9112085|Benign|0.085|0.085|-0.0889|0.0|
|9112366|Benign|0.1422|0.1422|-0.1534|0.0|
|9112367|Benign|0.0038|0.0038|-0.0038|0.0|
|9112594|Benign|0.0009|0.0009|-0.0009|0.0|
|9112712|Benign|0.0001|0.0001|-0.0001|0.0|
|911296201|Malignant|0.9998|0.0002|0.0002|0.0|
|911296202|Malignant|1.0|0.0|0.0|0.0|
|9113156|Benign|0.0018|0.0018|-0.0018|0.0|
|911320501|Benign|0.0002|0.0002|-0.0002|0.0|
|911320502|Benign|0.0003|0.0003|-0.0003|0.0|
|9113239|Benign|0.3452|0.3452|-0.4235|0.0|
|9113455|Benign|0.0478|0.0478|-0.049|0.0|
|9113514|Benign|0.0|0.0|0.0|0.0|
|9113538|Malignant|0.9999|0.0001|0.0001|0.0|
|911366|Benign|0.4296|0.4296|-0.5615|0.0|
|9113778|Benign|0.0001|0.0001|-0.0001|0.0|
|9113816|Benign|0.0065|0.0065|-0.0065|0.0|
|911384|Benign|0.0012|0.0012|-0.0012|0.0|
|9113846|Benign|0.0005|0.0005|-0.0005|0.0|
|911391|Benign|0.0003|0.0003|-0.0003|0.0|
|911408|Benign|0.0007|0.0007|-0.0007|0.0|
|911654|Benign|0.0321|0.0321|-0.0326|0.0|
|911673|Benign|0.0|0.0|0.0|0.0|
|911685|Benign|0.0003|0.0003|-0.0003|0.0|
|911916|Malignant|0.9955|0.0045|0.0045|0.0|
|912193|Benign|0.0002|0.0002|-0.0002|0.0|
|91227|Benign|0.0005|0.0005|-0.0005|0.0|
|912519|Benign|0.0034|0.0034|-0.0034|0.0|
|912558|Benign|0.0009|0.0009|-0.0009|0.0|
|912600|Benign|0.0279|0.0279|-0.0283|0.0|
|913063|Benign|0.2924|0.2924|-0.3459|0.0|
|913102|Benign|0.0008|0.0008|-0.0008|0.0|
|913505|Malignant|0.9999|0.0001|0.0001|0.0|
|913512|Benign|0.0024|0.0024|-0.0024|0.0|
|913535|Malignant|0.0556|0.9444|2.8894|1.0|
|91376701|Benign|0.0011|0.0011|-0.0011|0.0|
|91376702|Benign|0.0004|0.0004|-0.0004|0.0|
|914062|Malignant|0.9908|0.0092|0.0092|0.0|
|914101|Benign|0.0|0.0|0.0|0.0|
|914102|Benign|0.0002|0.0002|-0.0002|0.0|
|914333|Benign|0.0136|0.0136|-0.0137|0.0|
|914366|Benign|0.0822|0.0822|-0.0858|0.0|
|914580|Benign|0.0005|0.0005|-0.0005|0.0|
|914769|Malignant|0.9945|0.0055|0.0055|0.0|
|91485|Malignant|1.0|0.0|0.0|0.0|
|914862|Benign|0.0078|0.0078|-0.0078|0.0|
|91504|Malignant|0.9987|0.0013|0.0013|0.0|
|91505|Benign|0.0064|0.0064|-0.0065|0.0|
|915143|Malignant|1.0|0.0|0.0|0.0|
|915186|Benign|0.2845|0.2845|-0.3347|0.0|
|915276|Benign|0.1476|0.1476|-0.1597|0.0|
|91544001|Benign|0.0117|0.0117|-0.0117|0.0|
|91544002|Benign|0.007|0.007|-0.0071|0.0|
|915452|Benign|0.0069|0.0069|-0.0069|0.0|
|915460|Malignant|0.9999|0.0001|0.0001|0.0|
|91550|Benign|0.0001|0.0001|-0.0001|0.0|
|915664|Benign|0.0001|0.0001|-0.0001|0.0|
|915691|Malignant|0.9799|0.0201|0.0203|0.0|
|915940|Benign|0.0016|0.0016|-0.0016|0.0|
|91594602|Malignant|0.0331|0.9669|3.4087|1.0|
|916221|Benign|0.001|0.001|-0.001|0.0|
|916799|Malignant|0.9948|0.0052|0.0052|0.0|
|916838|Malignant|0.9982|0.0018|0.0018|0.0|
|917062|Benign|0.0639|0.0639|-0.066|0.0|
|917080|Benign|0.0076|0.0076|-0.0076|0.0|
|917092|Benign|0.0033|0.0033|-0.0033|0.0|
|91762702|Malignant|1.0|0.0|0.0|0.0|
|91789|Benign|0.0|0.0|0.0|0.0|
|917896|Benign|0.0401|0.0401|-0.0409|0.0|
|917897|Benign|0.0|0.0|0.0|0.0|
|91805|Benign|0.0|0.0|0.0|0.0|
|91813701|Benign|0.0798|0.0798|-0.0832|0.0|
|91813702|Benign|0.0|0.0|0.0|0.0|
|918192|Benign|0.0545|0.0545|-0.0561|0.0|
|918465|Benign|0.0002|0.0002|-0.0002|0.0|
|91858|Benign|0.0039|0.0039|-0.0039|0.0|
|91903901|Benign|0.0108|0.0108|-0.0109|0.0|
|91903902|Benign|0.0002|0.0002|-0.0002|0.0|
|91930402|Malignant|0.9984|0.0016|0.0016|0.0|
|919537|Benign|0.001|0.001|-0.001|0.0|
|919555|Malignant|0.9999|0.0001|0.0001|0.0|
|91979701|Malignant|0.7659|0.2341|0.2667|0.0|
|919812|Benign|0.8086|0.8086|-1.6536|1.0|
|921092|Benign|0.0|0.0|0.0|0.0|
|921362|Benign|0.0086|0.0086|-0.0086|0.0|
|921385|Benign|0.0004|0.0004|-0.0004|0.0|
|921386|Benign|0.7953|0.7953|-1.5862|1.0|
|921644|Benign|0.0411|0.0411|-0.042|0.0|
|922296|Benign|0.0085|0.0085|-0.0086|0.0|
|922297|Benign|0.0031|0.0031|-0.0031|0.0|
|922576|Benign|0.0049|0.0049|-0.0049|0.0|
|922577|Benign|0.0|0.0|0.0|0.0|
|922840|Benign|0.0001|0.0001|-0.0001|0.0|
|923169|Benign|0.0|0.0|0.0|0.0|
|923465|Benign|0.0007|0.0007|-0.0007|0.0|
|923748|Benign|0.0|0.0|0.0|0.0|
|923780|Benign|0.001|0.001|-0.001|0.0|
|924084|Benign|0.0047|0.0047|-0.0047|0.0|
|924342|Benign|0.0|0.0|0.0|0.0|
|924632|Benign|0.0127|0.0127|-0.0127|0.0|
|924934|Benign|0.0177|0.0177|-0.0179|0.0|
|924964|Benign|0.0|0.0|0.0|0.0|
|925236|Benign|0.0|0.0|0.0|0.0|
|925277|Benign|0.0498|0.0498|-0.0511|0.0|
|925291|Benign|0.0613|0.0613|-0.0633|0.0|
|925292|Benign|0.2555|0.2555|-0.2951|0.0|
|925311|Benign|0.0|0.0|0.0|0.0|
|925622|Malignant|1.0|0.0|0.0|0.0|
|926125|Malignant|1.0|0.0|0.0|0.0|
|926424|Malignant|1.0|0.0|0.0|0.0|
|926682|Malignant|0.9999|0.0001|0.0001|0.0|
|926954|Malignant|0.9088|0.0912|0.0957|0.0|
|927241|Malignant|1.0|0.0|0.0|0.0|
|92751|Benign|0.0|0.0|0.0|0.0|


