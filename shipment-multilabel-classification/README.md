# **Multilabel Classification for Predicting Shipment Modes**

## **Project Overview**
The transport industry plays a critical role in the global economy. Efficient movement of goods is essential for businesses to operate smoothly and for customers to receive their products on time. Determining the appropriate mode of transport for each shipment is a complex task that involves considering various factors such as the type of product, the distance, and the destination.

Choosing the right mode of transport significantly impacts delivery time, cost, and safety. For example, air transport is generally faster but more expensive, while sea transport is slower but more cost-effective for larger shipments. Incorrect choices in transport mode can lead to delays, damage to goods, or increased business costs. By accurately predicting the most suitable mode of transport, businesses can optimize their logistics, reduce costs, and enhance customer satisfaction.

## **Key Terms**
- **Shipment Modes**: Different methods of transportation, such as air, sea, and road.
- **Multilabel Classification**: A type of machine learning problem where each instance is assigned multiple labels, rather than just one.
- **Features**: Data points used to predict the transport mode, such as product type, distance, and destination.
  
## **Data Overview**
The dataset includes information on various shipments, such as:
- **Product Type**: The category of the product being shipped.
- **Distance**: The distance between the origin and destination.
- **Destination**: The geographical location to which the product is being transported.
- **Shipment Mode**: The transport method used for shipment (this is the target variable in the classification problem).

## **Objective**
The goal of this project is to predict the shipment mode using machine learning. I explore several multilabel classification approaches to determine the most appropriate transport mode for each shipment, which will help businesses optimize logistics and improve operational efficiency.

## **Libraries**
The following libraries are required to run the notebook:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `lightgbm`
- `sklearn`
- `keras`
- `tensorflow`
  
## **Approach**

### **Exploratory Data Analysis (EDA)**
- Understand the features and their relationships.
- Check the data summary for missing or invalid values.

### **Preprocessing**
- Encode categorical features.
- Split the dataset into training and testing sets.
  
### **Cross-validation Sets**
- Create cross-validation sets for model evaluation.

### **Multilabel Classification Approaches**
1. **Naive Independent Models**: 
    - Train separate binary classifiers for each target label (using LightGBM).
    - Evaluate the performance of each model using the F1 score.
   
2. **Classifier Chains**:
    - Train a binary classifier for each target label and chain them together.
    - Use the output of one classifier as input for the next.
    - Evaluate the performance using the F1 score.
  
3. **Natively Multilabel Models**:
    - Use models like Extra Trees and Neural Networks that can handle multiple labels directly.
    - Evaluate model performance with the F1 score.
  
4. **Multilabel to Multiclass Approach**:
    - Combine different combinations of labels into a single target label.
    - Train a classifier (e.g., LightGBM) on the combined labels.
    - Evaluate the model using metrics such as F1 score, precision, and recall.

## **Evaluation**
The models are evaluated using metrics such as F1 score, precision, recall, and accuracy. The best-performing model is selected based on these metrics.
