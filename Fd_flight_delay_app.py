
import streamlit as st
import numpy as np
import joblib

# Load saved model and encoders
loaded_model   = joblib.load("xgb_flight_delay_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")
feature_names  = joblib.load("feature_names.pkl")

def encode_value(col, value):
    le = label_encoders[col]
    if value in le.classes_:
        return le.transform([value])[0]
    else:
        return 0

def predict_delay(features):
    features_array = np.array(features)
    single_sample  = features_array.reshape(1, -1)
    probability    = loaded_model.predict_proba(single_sample)[0][1]
    prediction     = 1 if probability >= 0.35 else 0
    return prediction, probability

def main():
    st.title("✈️ Flight Arrival Delay Predictor")
    st.write("Enter flight details below to predict whether the flight will arrive 15+ minutes late.")

    # ── Input fields ──────────────────────────────────────────────────────────
    Year       = st.selectbox("Year", [2018, 2019, 2021, 2022])
    Quarter    = st.selectbox("Quarter", [1, 2, 3, 4])
    Month      = st.selectbox("Month", list(range(1, 13)))
    DayofMonth = st.slider("Day of Month", 1, 31, 15)
    DayOfWeek  = st.selectbox("Day of Week", [1, 2, 3, 4, 5, 6, 7],
                               format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x-1])

    Marketing_Airline = st.selectbox("Marketing Airline",
                                     label_encoders["Marketing_Airline_Network"].classes_.tolist())
    Operating_Airline = st.selectbox("Operating Airline",
                                     label_encoders["Operating_Airline"].classes_.tolist())
    Origin        = st.selectbox("Origin Airport", label_encoders["Origin"].classes_.tolist())
    Dest          = st.selectbox("Destination Airport", label_encoders["Dest"].classes_.tolist())
    Distance      = st.number_input("Distance (miles)", min_value=16, max_value=5812, value=500)
    DistanceGroup = st.selectbox("Distance Group", list(range(1, 12)))
    CRSDepTime    = st.number_input("Scheduled Departure Time (hhmm)", min_value=1, max_value=2359, value=800)
    CRSArrTime    = st.number_input("Scheduled Arrival Time (hhmm)", min_value=1, max_value=2400, value=1100)
    DepTimeBlk    = st.selectbox("Departure Time Block",
                                  label_encoders["DepTimeBlk"].classes_.tolist())

    st.markdown("---")
    st.subheader("Airport Congestion Estimates")
    st.caption("Enter the number of flights scheduled at the same airport in the same departure/arrival hour.")
    OriginHourlyDepartures = st.number_input("Flights departing same origin in same hour",
                                              min_value=1, max_value=70, value=10)
    DestHourlyArrivals     = st.number_input("Flights arriving same destination in same hour",
                                              min_value=1, max_value=70, value=10)

    # ── Encode and assemble features ─────────────────────────────────────────
    features = [
        Year, Quarter, Month, DayofMonth, DayOfWeek,
        encode_value("Marketing_Airline_Network", Marketing_Airline),
        encode_value("Operating_Airline", Operating_Airline),
        encode_value("Origin", Origin),
        encode_value("Dest", Dest),
        Distance, DistanceGroup,
        CRSDepTime, CRSArrTime,
        encode_value("DepTimeBlk", DepTimeBlk),
        OriginHourlyDepartures,
        DestHourlyArrivals
    ]

    # ── Predict ───────────────────────────────────────────────────────────────
    if st.button("Predict"):
        prediction, probability = predict_delay(features)
        st.write(" ")
        if prediction == 1:
            st.error("⚠️  DELAYED — This flight is likely to arrive 15+ minutes late.")
        else:
            st.success("✅  ON TIME — This flight is expected to arrive on time.")
        st.write(f"Delay probability: **{probability:.1%}**")
        st.caption("Threshold: flights with probability ≥ 35% are classified as delayed.")

if __name__ == "__main__":
    main()
