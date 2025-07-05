import cv2
import os
from flask import Flask, request, render_template, redirect, url_for
from datetime import date, datetime, timedelta
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical

app = Flask(__name__)

nimgs = 20
# Load the background image from the static folder with error handling
background_image_path = None
for img_name in ['background.png', 'background.jpg']:
    img_path = os.path.join('static', img_name)
    if os.path.exists(img_path):
        background_image_path = img_path
        break

# Define target dimensions for the frame
TARGET_HEIGHT = 480
TARGET_WIDTH = 640

if background_image_path:
    imgBackground = cv2.imread(background_image_path)
    if imgBackground is None:
        print(f"Error: {background_image_path} could not be loaded. Using a blank image as fallback.")
        imgBackground = np.zeros((TARGET_HEIGHT + 162, TARGET_WIDTH + 55, 3), dtype=np.uint8)
    else:
        # Resize background to ensure it can accommodate the frame
        if imgBackground.shape[0] < TARGET_HEIGHT + 162 or imgBackground.shape[1] < TARGET_WIDTH + 55:
            imgBackground = cv2.resize(imgBackground, (TARGET_WIDTH + 55, TARGET_HEIGHT + 162))
else:
    print("Error: Neither background.png nor background.jpg found. Using a blank image as fallback.")
    imgBackground = np.zeros((TARGET_HEIGHT + 162, TARGET_WIDTH + 55, 3), dtype=np.uint8)

datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Create necessary directories and files
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
        f.write('Name,Roll Number,Entry Time,Exit Time,Re-Entry Time,Duration (seconds),Period\n')

# Global variables to track periods
current_period = 1
period_start_time = datetime.now()
period_duration = timedelta(minutes=2)

last_user = None
last_time = None

user_presence = {}

def totalreg():
    #Counts how many users are registered by checking how many folders are inside static/faces (each user has their own folder)
    return len(os.listdir('static/faces'))

def extract_faces(img):
    #Detects faces from an input image (img) using Haar Cascade
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))
        return face_points
    except Exception as e:
        print(f"Error in extract_faces: {e}")
        return []

def identify_face(facearray):
    model = tf.keras.models.load_model('static/face_recognition_model.h5')
    facearray = facearray.reshape(1, 50, 50, 3) / 255.0
    predictions = model.predict(facearray)
    userlist = os.listdir('static/faces')
    
    if len(userlist) == 0:
        print("No registered users found.")
        return None
    
    predicted_user_index = np.argmax(predictions)
    
    if predicted_user_index >= len(userlist):
        print("Invalid prediction index:", predicted_user_index)
        return None
    
    return userlist[predicted_user_index]

def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    label_dict = {user: i for i, user in enumerate(userlist)}

    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face)
            labels.append(label_dict[user])

    faces = np.array(faces) / 255.0
    labels = np.array(labels)
    labels = to_categorical(labels)

    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(50, 50, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(len(userlist), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(faces, labels, epochs=10, batch_size=32)
    model.save('static/face_recognition_model.h5')

def extract_attendance():
    try:
        df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
        print("DataFrame contents:")
        print(df.head())
        df.columns = df.columns.str.strip()
        
        if df.empty:
            return [], [], [], [], [], [], [], 0, []
        
        if 'Period' not in df.columns:
            raise KeyError("The 'Period' column is missing from the attendance file.")

        names = df['Name']
        rolls = df['Roll Number']
        entry_times = df['Entry Time']
        exit_times = df['Exit Time']
        re_entry_times = df['Re-Entry Time']
        durations = df['Duration (seconds)']
        periods = df['Period']
        l = len(df)
        
        return names, rolls, entry_times, exit_times, re_entry_times, durations, periods, l, entry_times.tolist()
    except Exception as e:
        print(f"Error reading attendance file: {e}")
        return [], [], [], [], [], [], [], 0, []

def is_session_ended():
    first_period_start_time = datetime.strptime("11:07:00", "%H:%M:%S").time()
    last_period_end_time = (datetime.combine(date.today(), first_period_start_time) + timedelta(minutes=14)).time()
    current_time_obj = datetime.now().time()
    # Session is ended if current time is before start or after end
    return current_time_obj < first_period_start_time or current_time_obj > last_period_end_time

def add_attendance(name):
    global current_period, last_user, last_time, user_presence
    if is_session_ended():
        return "Today's attendance has been successfully captured"
    
    print(f"Adding attendance for: {name}")
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")

    first_period_start_time = datetime.strptime("11:07:00", "%H:%M:%S").time()
    time_difference = datetime.combine(date.today(), datetime.now().time()) - datetime.combine(date.today(), first_period_start_time)
    current_period = (time_difference.seconds // 120) + 1
    if current_period < 1:
        current_period = 1
    if current_period > 7:
        current_period = 7

    print(f"Current period: {current_period}, User ID: {userid}")

    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    user_present = df[(df['Roll Number'] == int(userid)) & (df['Period'] == current_period)]
    
    if not user_present.empty:
        if pd.isna(user_present['Exit Time'].iloc[0]):
            print(f"User {userid} already marked present for period {current_period}.")
            return
        else:
            df.loc[(df['Roll Number'] == int(userid)) & (df['Period'] == current_period), 'Re-Entry Time'] = current_time
            print(f"User {userid} re-entered at {current_time} for period {current_period}.")
    else:
        new_record = pd.DataFrame({
            'Name': [username],
            'Roll Number': [int(userid)],
            'Entry Time': [current_time],
            'Exit Time': [None],
            'Re-Entry Time': [None],
            'Duration (seconds)': [None],
            'Period': [current_period]
        })
        df = pd.concat([df, new_record], ignore_index=True)
        print(f"Attendance recorded for {username} ({userid}) at {current_time} for period {current_period}.")

    df.to_csv(f'Attendance/Attendance-{datetoday}.csv', index=False)
    last_user = userid
    last_time = datetime.now()
    user_presence[userid] = {'entry_time': current_time, 'exit_time': None}

def update_exit_time(userid):
    global user_presence
    if is_session_ended():
        return
    
    current_time = datetime.now().strftime("%H:%M:%S")
    
    if userid in user_presence:
        user_presence[userid]['exit_time'] = current_time
        df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
        user_row = df[(df['Roll Number'] == int(userid)) & (df['Exit Time'].isnull())]

        if not user_row.empty:
            df.loc[user_row.index, 'Exit Time'] = current_time
            entry_time = user_row['Entry Time'].iloc[0]
            entry_time_dt = datetime.strptime(entry_time, "%H:%M:%S")
            exit_time_dt = datetime.strptime(current_time, "%H:%M:%S")
            duration = (exit_time_dt - entry_time_dt).total_seconds()
            df.loc[user_row.index, 'Duration (seconds)'] = duration
            df.to_csv(f'Attendance/Attendance-{datetoday}.csv', index=False)
            print(f"Exit time updated for user {userid} at {current_time}. Duration: {duration} seconds.")
        else:
            print(f"No entry found for user {userid} to update exit time.")

def getallusers():
    userlist = os.listdir('static/faces')
    names = []
    rolls = []
    l = len(userlist)

    for i in userlist:
        name, roll = i.split('_')
        names.append(name)
        rolls.append(roll)

    return userlist, names, rolls, l

@app.route('/')
def home():
    names, rolls, entry_times, exit_times, re_entry_times, durations, periods, l, times = extract_attendance()
    session_ended = is_session_ended()
    message = "Today's attendance has been successfully captured" if session_ended else None
    return render_template('home.html', names=names, rolls=rolls, entry_times=entry_times, exit_times=exit_times, 
                           re_entry_times=re_entry_times, durations=durations, periods=periods, l=l, 
                           totalreg=totalreg(), datetoday2=datetoday2, current_period=current_period, 
                           session_ended=session_ended, mess=message, times=times)

@app.route('/new-user', methods=['GET'])
def new_user():
    session_ended = is_session_ended()
    if session_ended:
        return render_template('home.html', mess="Today's attendance has been successfully captured", totalreg=totalreg())
    return render_template('new_user.html', totalreg=totalreg())

@app.route('/attendance')
def attendance():
    names, rolls, entry_times, exit_times, re_entry_times, durations, periods, l, times = extract_attendance()
    session_ended = is_session_ended()
    message = "Today's attendance has been successfully captured" if session_ended else None
    return render_template('attendance.html', names=names, rolls=rolls, entry_times=entry_times, periods=periods, 
                           l=l, totalreg=totalreg(), datetoday2=datetoday2, current_period=current_period, 
                           session_ended=session_ended, mess=message, times=times)

@app.route('/start', methods=['GET'])
def start():
    names, rolls, entry_times, exit_times, re_entry_times, durations, periods, l, times = extract_attendance()
    if is_session_ended():
        return redirect(url_for('attendance', mess="Today's attendance has been successfully captured"))

    if 'face_recognition_model.h5' not in os.listdir('static'):
        return render_template('home.html', names=names, rolls=rolls, entry_times=entry_times, exit_times=exit_times, 
                               re_entry_times=re_entry_times, durations=durations, periods=periods, l=l, 
                               totalreg=totalreg(), datetoday2=datetoday2, 
                               mess='There is no trained model in the static folder. Please add a new face to continue.', times=times)

    ret = True
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)
    detected_users = set()

    while ret:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Failed to capture frame from webcam.")
            break

        # Resize frame to match the target region
        frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))
        
        # Extract faces from the frame
        faces = extract_faces(frame)
        current_detected_users = set()

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (86, 32, 251), 1)
            cv2.rectangle(frame, (x, y), (x+w, y-40), (86, 32, 251), -1)
            face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
            identified_person = identify_face(face)
            if identified_person is not None:
                userid = identified_person.split('_')[1]
                current_detected_users.add(userid)
                if userid not in detected_users:
                    result = add_attendance(identified_person)
                    if result == "Today's attendance has been successfully captured":
                        cap.release()
                        cv2.destroyAllWindows()
                        return redirect(url_for('attendance', mess="Today's attendance has been successfully captured"))
                    detected_users.add(userid)
                cv2.putText(frame, f'{identified_person}', (x, y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)

        for userid in detected_users - current_detected_users:
            update_exit_time(userid)
            detected_users.remove(userid)

        # Place frame in background
        imgBackground[162:162 + TARGET_HEIGHT, 55:55 + TARGET_WIDTH] = frame
        cv2.imshow('Attendance', imgBackground)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    names, rolls, entry_times, exit_times, re_entry_times, durations, periods, l, times = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, entry_times=entry_times, exit_times=exit_times, 
                           re_entry_times=re_entry_times, durations=durations, periods=periods, l=l, 
                           totalreg=totalreg(), datetoday2=datetoday2, current_period=current_period, times=times)

@app.route('/add', methods=['GET', 'POST'])
def add():
    if is_session_ended():
        return render_template('home.html', mess="Today's attendance has been successfully captured", totalreg=totalreg())
    
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    userimagefolder = 'static/faces/'+newusername+'_'+str(newuserid)
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    i, j = 0, 0
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)
    while 1:
        _, frame = cap.read()
        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
            cv2.putText(frame, f'Images Captured: {i}/{nimgs}', (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
            if j % 5 == 0:
                name = newusername+'_'+str(i)+'.jpg'
                cv2.imwrite(userimagefolder+'/'+name, frame[y:y+h, x:x+w])
                i += 1
            j += 1
        if j == nimgs*5:
            break
        cv2.imshow('Adding new User', frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    print('Training Model')
    train_model()
    
    names, rolls, entry_times, exit_times, re_entry_times, durations, periods, l, times = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, entry_times=entry_times, exit_times=exit_times, 
                           re_entry_times=re_entry_times, durations=durations, periods=periods, l=l, 
                           totalreg=totalreg(), datetoday2=datetoday2, current_period=current_period, times=times)

if __name__ == '__main__':
    app.run(debug=True)