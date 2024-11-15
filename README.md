# Wysa_NLP_Engineer_Assignment

### Working Prototype: https://emotion-and-brand-recognition.streamlit.app/

---
### High Level Overview 
![image](https://github.com/user-attachments/assets/1abfeef8-d0fd-4481-b99e-3084f09141bf)


---
### Ways to Improve the Dataset Further:
1. **Creating more granular emotion categories**
> Instead of only incorporating broad categories like “Positive emotion” and “Negative emotion” in the dataset, we can include more categories for specific emotions such as “Excitement”, “Frustration”, “Anticipation”, “Joy”, “Disappointment” etc. This will provide a more clear and nuanced understanding of the user sentiment.

2. **Supporting Multi-label Classification**
> Instead of associating a tweet with a single emotion, we can have a tweet associated with multiple emotion categories. This is because some tweets can express mix of feelings e.g., both "Excitement" and "Frustration" together. This will reflect real-world complexity.

3. **Adding more samples to the Dataset**
> Having a larger dataset would ensure a better coverage of emotions and brand mentions, improving model generalization. It should also be ensured that samples for various classes should be evenly distributed.

4. **Include Geolocation Data**
> If available, location data can be integrated to understand regional sentiment variations. This can be particularly useful for location-based brand sentiment analysis.

---
### Approach to Evaluate the Trained Model:
**Confidence-based Analysis**
> We will analyze the probability (confidence) scores of the model predictions for the tweets in the test set. This will give us some insight into model certainty and can help identify classes the model is/isn't confident about. High confidence scores generally indicate more reliable predictions.
