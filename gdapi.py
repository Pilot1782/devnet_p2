import json
import os
import flask
import requests
from datetime import datetime

import google_auth_oauthlib.flow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# This variable specifies the name of a file that contains the OAuth 2.0
# information for this application, including its client_id and client_secret.
CLIENT_SECRETS_FILE = "clientSecret.json"

# This OAuth 2.0 access scope allows for full read/write access to the
# authenticated user's account and requires requests to use an SSL connection.
SCOPES = ['https://www.googleapis.com/auth/drive.metadata.readonly']
API_SERVICE_NAME = 'drive'
API_VERSION = 'v2'

app = flask.Flask(__name__)
# Note: A secret key is included in the sample so that it works.
# If you use this code in your application, replace this with a truly secret
# key. See https://flask.palletsprojects.com/quickstart/#sessions.
app.secret_key = 'bazingaaosmrvfpeanmgaaogmaeo[gmadgsdfasdfsafsgfsg]'

current_credentials = None
service = None
lastImageData = { "id": "" }

def saveData():
  #json_LastImgData = json.dumps(lastImageData)
  global lastImageData, current_credentials
  credAndLastImgData = { 'creds': current_credentials, 'imgData':  lastImageData }
  with open('data.json', 'w') as data:
    #current cred, lastimagedata
    data.seek(0)
    json.dump(credAndLastImgData, data)
    data.truncate()

def loadData():
  global lastImageData, current_credentials
  with open('data.json', 'r') as data:
    newData = json.load(data)
    current_credentials = newData['creds']
    lastImageData = newData['imgData']

@app.route('/')
def index():
  return print_index_table()


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
      access_type='offline',
      # Enable incremental authorization. Recommended as a best practice.
      include_granted_scopes='true')

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
  current_credentials = flow.credentials
  #saveData()

  global service
  service = build("drive", "v3", credentials=current_credentials)

  return flask.redirect("/test")#flask.url_for('test_api_request'))


@app.route('/revoke')
def revoke():
  global current_credentials
  if current_credentials is not None:
    current_credentials = None
  else:
    return('No credentials to revoke.' + print_index_table())

  revoke = requests.post('https://oauth2.googleapis.com/revoke',
      params={'token': current_credentials.token},
      headers = {'content-type': 'application/x-www-form-urlencoded'})

  status_code = getattr(revoke, 'status_code')
  if status_code == 200:
    return('Credentials successfully revoked.' + print_index_table())
  else:
    return('An error occurred.' + print_index_table())


@app.route('/clear')
def clear_credentials():
  if 'credentials' in flask.session:
    del flask.session['credentials']
  return ('Credentials have been cleared.<br><br>' +
          print_index_table())


def credentials_to_dict(credentials):
  return {'token': credentials.token,
          'refresh_token': credentials.refresh_token,
          'token_uri': credentials.token_uri,
          'client_id': credentials.client_id,
          'client_secret': credentials.client_secret,
          'scopes': credentials.scopes}

def print_index_table():
  return ('<table>' +
          '<tr><td><a href="/test">Test an API request</a></td>' +
          '<td>Submit an API request and see a formatted JSON response. ' +
          '    Go through the authorization flow if there are no stored ' +
          '    credentials for the user.</td></tr>' +
          '<tr><td><a href="/authorize">Test the auth flow directly</a></td>' +
          '<td>Go directly to the authorization flow. If there are stored ' +
          '    credentials, you still might not be prompted to reauthorize ' +
          '    the application.</td></tr>' +
          '<tr><td><a href="/revoke">Revoke current credentials</a></td>' +
          '<td>Revoke the access token associated with the current user ' +
          '    session. After revoking credentials, if you go to the test ' +
          '    page, you should see an <code>invalid_grant</code> error.' +
          '</td></tr>' +
          '<tr><td><a href="/clear">Clear Flask session credentials</a></td>' +
          '<td>Clear the access token currently stored in the user session. ' +
          '    After clearing the token, if you <a href="/test">test the ' +
          '    API request</a> again, you should go back to the auth flow.' +
          '</td></tr></table>')

@app.route("/test")
def getNewImage():
  results = service.files().list(
      q="'1PNm562W_IqKJ8Zxl8bz03p_yiEZoh88W' in parents", spaces="drive", orderBy="name_natural desc", fields="nextPageToken, files(id, modifiedTime, name)"
    ).execute()
# all files from specific folder name with link
  items = results.get('files', [])
  newestImg = items[0]

  global lastImageData
  newestImg["prevRecieved"] = lastImageData["id"] == newestImg["id"]
  lastImageData = newestImg
  #saveData()
  #modifiedTime = datetime.strptime(items[0]["modifiedTime"][:-5], "%Y-%m-%dT%H:%M:%S")
  return flask.redirect("/view")#newestImg #TODO:revert
#str(modifiedTime.timestamp() > lastCreationDate)
  
  #Retrieve a list of all files from google drive (files.list method)
  #Get the most recent creation date
  #Compare against a variable "lastcreationdate"

@app.route("/view")
def viewCurrentData():
  #TODO: IsHealthy is a boolean indicating whether the plant is healthy or not healthy.
  return flask.render_template("view.html", fileid=lastImageData["id"], filename=lastImageData["name"], isHealthy=True)

@app.route("/trainModel")
def trainModel():
  pass



if __name__ == '__main__':
  # When running locally, disable OAuthlib's HTTPs verification.
  # ACTION ITEM for developers:
  #     When running in production *do not* leave this option enabled.
  os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

  #loadData()

  # Specify a hostname and port that are set as a valid redirect URI
  # for your API project in the Google API Console.
  app.run('localhost', 8080, debug=True)

