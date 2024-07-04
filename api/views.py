from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import numpy as np
import pandas as pd
import tensorflow as tf

model = tf.keras.models.load_model("prediction_model/diabetes_model.h5")

@csrf_exempt
def getdata(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)

            pregnancies = float(data.get('pregnancies', 0))
            glucose = float(data.get('glucose', 0))
            blood_pressure = float(data.get('blood_pressure', 0))
            skin_thickness = float(data.get('skin_thickness', 0))
            insulin = float(data.get('insulin', 0))
            bmi = float(data.get('bmi', 0))
            diabetes_pedigree_function = float(data.get('diabetes_pedigree_function', 0))
            age = float(data.get('age', 0))

            input_data = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]],
                                      columns=['pregnancies', 'glucose', 'blood_pressure', 'skin_thickness', 'insulin', 'bmi', 'diabetes_pedigree_function', 'age'])

            input_data_array = input_data.to_numpy()

            processed_data = model.predict(input_data_array)

            prediction_result = processed_data[0][0] if len(processed_data[0]) == 1 else processed_data[0]

            # Send the processed data back to the client
            return JsonResponse({'predictionResult': float(prediction_result)})
        
        except Exception as e:
            return JsonResponse({'error': str(e)})

    return JsonResponse({'error': 'Invalid request method'})
