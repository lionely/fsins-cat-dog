import requests 

# https://your-heroku-app-name.herokuapp.com/predict
# http://localhost:5000/predict

#source of test image https://www.pixilart.com/art/32-x-32-cat-42e019c236fe77a
response = requests.post("http://localhost:5000/predict", files={'file': open('cat.png', 'rb')})

print(response.text)