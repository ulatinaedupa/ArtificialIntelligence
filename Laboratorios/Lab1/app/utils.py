import numpy as np
import pandas as pd

# just storing all the others transformations to keep the prediction
def transform_data(payload, col_order_out, mean_eve_min, ohe):
    payload_ohe = ohe.transform(payload[ohe.feature_names_in_]).toarray()
    payload_ohe = pd.DataFrame(payload_ohe, columns=ohe.get_feature_names_out())
    payload_concat = pd.concat([payload, payload_ohe], axis=1)
    payload_concat['log_number_customer_service_calls'] = np.log1p(payload_concat['number_customer_service_calls'])
    payload_concat['day_ratio'] = payload_concat['total_day_charge'] / payload_concat['total_day_minutes']
    payload_concat['eve_ratio'] = payload_concat['total_eve_charge'] / payload_concat['total_eve_minutes']
    payload_concat['night_ratio'] = payload_concat['total_night_charge'] / payload_concat['total_night_minutes']
    payload_concat['intl_ratio'] = payload_concat['total_intl_charge'] / payload_concat['total_intl_minutes']
    payload_concat['unhappy_customers'] = ((payload_concat['remaining_term'] < 5) & 
                                           (payload_concat['last_nps_rating'] <= 7) & 
                                           (payload_concat.promotions_offered == 'No')).astype(bool)
    payload_concat['total_eve_minutes_missing'] = payload_concat.total_eve_minutes.isna().astype(bool)
    payload_concat.total_eve_minutes.fillna(mean_eve_min, inplace=True) 
    
    payload_transformed = payload_concat[col_order_out]
    return payload_transformed