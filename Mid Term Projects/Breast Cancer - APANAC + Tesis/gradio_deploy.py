import gradio as gr
import keras
import numpy as np

def f1(y_true, y_pred):
    y_pred = K.round(y_pred)
    # Calculate true positives, true negatives, false positives and false negatives
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    # Calculate precision and recall
    # Adding epsilon (small value) to combat potential division by 0
    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    # Calculate F1
    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)

model_tuned = keras.models.load_model('breast_cancer_effnet_tuned.h5', custom_objects={'f1':f1})
labels = ['No Breast Cancer', 'Breast Cancer']

def predict(inp):
        image = np.expand_dims(inp, 0)
        pred = model_tuned(image)
        prediction = np.squeeze(pred)
        confidences = {labels[1*(prediction > 0.5)]: float(prediction)}
        return confidences

demo = gr.Interface(fn=predict, 
             inputs=gr.inputs.Image(type="pil"),
             outputs=gr.outputs.Label(num_top_classes=2),
             examples=[["example/batch_sample_0.png"], ["example/batch_sample_1.png"]],
             )
             
demo.launch(share=True)