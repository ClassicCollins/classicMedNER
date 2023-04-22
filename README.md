# classicMedNER
Named Entity Recognition (NER)
NER is the process of classifying entities into predefined categories such as person, date, time, location, organisation, proffession etC
This particular NER is developed and customized for medical records.The task is to deploy a named extraction pipeline using Tensorflow/keras.
The target for the NER pipeline is to identify and etract Type element in each tag across the records.
For example. "Vicente Blair, M.D., Yovani Vergara and their escort  visited Doctors Hospital North in Georgia to treat VALDEZ, Harlan" 
Vincente Blair - B-Doctor
Yovani Vergara - I-b-doctor
and - O
their -O
escort -O
Visited - O
Doctors Hospital North - B-Hospital
in - O
Georgia - B-State
to - O
treat - O
VALDEZ, Harlan - B-Patience

TOOLS and LIBRARIES USED:
Tensorflow.karas
Pandas
Streamlite

To learn more about this App kindly visit https://classicmedner.streamlit.app/
or Contact the Author on 08037953669 or ugwuozorcollinsezie@gmail.com
