import json
import os
import flask
import io
from datetime import datetime
from PIL import Image
import numpy as np

from HealthModel import HealthModel

import google_auth_oauthlib.flow
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload

# This variable specifies the name of a file that contains the OAuth 2.0
# information for this application, including its client_id and client_secret.
CLIENT_SECRETS_FILE = "clientSecret.json"

# This OAuth 2.0 access scope allows for full read/write access to the
# authenticated user's account and requires requests to use an SSL connection.
SCOPES = ['https://www.googleapis.com/auth/drive.metadata.readonly','https://www.googleapis.com/auth/drive.readonly']
API_SERVICE_NAME = 'drive'
API_VERSION = 'v2'

app = flask.Flask(__name__)
# Note: A secret key is included in the sample so that it works.
# If you use this code in your application, replace this with a truly secret
# key. See https://flask.palletsprojects.com/quickstart/#sessions.
app.secret_key = 'bazingaaosmrvfpeanmgaaogmaeo[gmadgsdfasdfsafsgfsg]'

current_credentials = None
service = None
selectedImageData = None
imageDataHistory = []

def credentials_to_dict(credentials):
  return {'token': credentials.token,
          'refresh_token': credentials.refresh_token,
          'token_uri': credentials.token_uri,
          'client_id': credentials.client_id,
          'client_secret': credentials.client_secret,
          'scopes': credentials.scopes}

def saveData():
  creds = credentials_to_dict(current_credentials)
  with open('userdata.json', 'w') as data:
    data.seek(0)
    json.dump(creds, data)
    data.truncate()

def saveUserData():
  with open('data.json', 'w') as data:
    data.seek(0)
    json.dump(imageDataHistory, data)
    data.truncate()

def loadUserData():
  try:
    with open('userdata.json', 'r') as data:
      newData = json.load(data)
      global current_credentials, service
      current_credentials = Credentials.from_authorized_user_info(info={'refresh_token': newData['refresh_token'],'client_id': newData['client_id'],'client_secret': newData['client_secret'],'token_uri': newData['token_uri']}, scopes=SCOPES)
      service = build("drive", "v3", credentials=current_credentials)
  except:
    pass

def loadData():
  try:
    with open('data.json', 'r') as data:
      newData = json.load(data)
      global imageDataHistory
      imageDataHistory = newData
  except:
    pass

loadUserData()
loadData()

def downloadImg(file_id):
  request = service.files().get_media(fileId=file_id)#, mimeType='image/jpg')

  fh = io.BytesIO()
  downloader = MediaIoBaseDownload(fh, request)
  done = False
  while done is False:
    status, done = downloader.next_chunk()
    print("Downloading " + file_id + ": %d%%" % int(status.progress() * 100))

  imageStream = Image.open(fh)
  imageStream = np.array(imageStream, dtype=np.uint8)
  print(imageStream.shape)
  return imageStream

@app.route('/')
def index():
  if current_credentials is not None: 
    return flask.redirect("/runmodel")
  else:
    return flask.redirect("/authorize")


#@app.route('/test')
#def test_api_request():
#  return flask.status(501)


@app.route('/authorize')
def authorize():
  if current_credentials is not None:
    return flask.status(423)
  # Create flow instance to manage the OAuth 2.0 Authorization Grant Flow steps.
  flow = google_auth_oauthlib.flow.Flow.from_client_secrets_file(
      CLIENT_SECRETS_FILE, scopes=SCOPES)

  # The URI created here must exactly match one of the authorized redirect URIs
  # for the OAuth 2.0 client, which you configured in the API Console. If this
  # value doesn't match an authorized URI, you will get a 'redirect_uri_mismatch'
  # error.
  flow.redirect_uri = flask.url_for('oauth2callback', _external=True)

  authorization_url, state = flow.authorization_url(
      # Enable offline access so that you can refresh an access token without
      # re-prompting the user for permission. Recommended for web server apps.
      access_type='offline')

  # Store the state so the callback can verify the auth server response.
  flask.session['state'] = state

  return flask.redirect(authorization_url)


@app.route('/oauth2callback')
def oauth2callback():
  # Specify the state when creating the flow in the callback so that it can
  # verified in the authorization server response.
  state = flask.session['state']

  flow = google_auth_oauthlib.flow.Flow.from_client_secrets_file(
      CLIENT_SECRETS_FILE, scopes=SCOPES, state=state)
  flow.redirect_uri = flask.url_for('oauth2callback', _external=True)

  # Use the authorization server's response to fetch the OAuth 2.0 tokens.
  authorization_response = flask.request.url
  flow.fetch_token(authorization_response=authorization_response)

  # Store credentials in the session.
  # ACTION ITEM: In a production app, you likely want to save these
  #              credentials in a persistent database instead.
  global current_credentials
  current_credentials = flow.credentials
  saveUserData()

  global service
  service = build("drive", "v3", credentials=current_credentials)

  return flask.redirect("/runmodel")#flask.url_for('test_api_request'))

@app.route("/runmodel")
def runNewImage():
  if current_credentials is None:
    return flask.redirect("/authorize")
  
  results = service.files().list(
      q="'1PNm562W_IqKJ8Zxl8bz03p_yiEZoh88W' in parents", spaces="drive", orderBy="name_natural desc", fields="nextPageToken, files(id, modifiedTime, name)"
    ).execute()
  
  items = results.get('files', [])
  newestImg = items[0]

  global imageDataHistory, selectedImageData

  for imgData in imageDataHistory:
    if imgData["id"] == newestImg["id"]:
      selectedImageData = imgData
      return flask.redirect("/view") #image has already been processed, load view with old data
  
  imageStream = downloadImg(newestImg["id"])
  model = HealthModel(os.path.join(os.getcwd(),"model.pth"))

  prediction = model.predict(imageStream)

  newestImg["runTime"] = datetime.now().isoformat()
  newestImg["healthy"] = bool(prediction[0])
  newestImg["confidence"] = prediction[1]

  imageDataHistory.push(newestImg)
  selectedImageData = newestImg
  saveData()

  return flask.redirect("/view")

@app.route("/view")
def viewCurrentData():
  if current_credentials is None:
    return flask.redirect("/authorize")
  
  return flask.render_template("view.html", fileid=selectedImageData["id"], filename=selectedImageData["name"], isHealthy=selectedImageData["healthy"], confidence=selectedImageData["confidence"], runTime=selectedImageData["runTime"])

@app.route("/trainmodel")
def trainModel():
  if current_credentials is None:
    return flask.redirect("/authorize")
  
  fileId = flask.request.args.get("FILEID")

  if fileId is None:
    return "", 400
  
  return "", 501

@app.route("/error")
def error():
  return "An error has occured.", 500

if __name__ == '__main__':
  # When running locally, disable OAuthlib's HTTPs verification.
  # ACTION ITEM for developers:
  #     When running in production *do not* leave this option enabled.
  os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

  #loadData()

  # Specify a hostname and port that are set as a valid redirect URI
  # for your API project in the Google API Console.
  app.run('localhost', 8080, debug=True)

