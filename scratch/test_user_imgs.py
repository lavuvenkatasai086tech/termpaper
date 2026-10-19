import glob, os, requests

url = 'http://127.0.0.1:5000/api/predict'
files = glob.glob('C:/Users/lavuv/.gemini/antigravity-ide/brain/3eb56759-7ef0-444b-9936-e2613c88e2bd/.user_uploaded/*.*')

print(f"Testing {len(files)} user uploaded images against live server:")
for f in sorted(files):
    with open(f, 'rb') as fp:
        res = requests.post(url, files={'file': fp})
    data = res.json()
    fname = os.path.basename(f)
    print(f"File: {fname} -> is_ood: {data.get('is_ood')} | pred: {data.get('predicted_class')} | err: {data.get('error_message')}")
