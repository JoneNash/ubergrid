wget https://archive.ics.uci.edu/ml/machine-learning-databases/blood-transfusion/transfusion.data
echo "months_since_donation,number_of_donations,donation_amount,time_since_first_donation,donated" > donations.csv
tail -n +2 transfusion.data >> donations.csv

