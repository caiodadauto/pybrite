import os
import zipfile
import argparse
from tqdm import tqdm
from pathlib import Path

from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

NAME = "topologies"

def unzipdir(zip_path):
  zipf = zipfile.ZipFile(zip_path, 'r', zipfile.ZIP_DEFLATED)
  zipf.extractall(zip_path.parent)

def zipdir(path, zip_path):
  zipf = zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED)
  for root, dirs, files in tqdm(os.walk(path)):
    for file in files:
      new_root = os.path.join(NAME, root.split("/")[-1])
      zipf.write(os.path.join(root, file), os.path.join(new_root, file))

def getid(drive, id_folder):
  zip_list={}
  file_list = drive.ListFile({'q': "'{}' in parents and trashed=false".format(id_folder)}).GetList()
  for f in file_list:
    if f['mimeType'] == 'application/zip':
      zip_list[f['title']] = f['id']
  return zip_list[NAME + ".zip"]

def logdrive():
  gauth = GoogleAuth()
  gauth.LoadCredentialsFile("mycreds.txt")
  if gauth.credentials is None:
    gauth.LocalWebserverAuth()
  elif gauth.access_token_expired:
    gauth.Refresh()
  else:
    gauth.Authorize()
  gauth.SaveCredentialsFile("mycreds.txt")
  drive = GoogleDrive(gauth)
  return drive

if __name__ == '__main__':
  p = argparse.ArgumentParser()
  p.add_argument("path", type=str, help="Path for Dataset")
  p.add_argument("--id-folder", type=str, default='1DEHJZQC6AFoolUeQqC6NnwPVg0RFZ8iK', help="Folder in drive to store dataset")
  p.add_argument("--download", action="store_true", help="Make download of Dataset")
  args = p.parse_args()

  print("Connecting to Drive...")
  drive = logdrive()
  print("Done!")

  dvc_path = Path(args.path)
  if args.download:
    zip_path = dvc_path.absolute().joinpath(NAME + ".zip")
    file_drive = drive.CreateFile({"id": getid(drive, args.id_folder)})
    print("Downloading " + NAME + ".zip" + " to " + str(zip_path.parent) + "...")
    file_drive.GetContentFile(zip_path)
    print("Done!")
    print("Unzip " + NAME + ".zip" + "...")
    unzipdir(zip_path)
    print("Done!")
  else:
    zip_path = dvc_path.absolute().parent.joinpath(NAME + ".zip")
    file_drive = drive.CreateFile(
      {"title": NAME + ".zip", "parents": [{"kind": "drive#fileLink", "id": args.id_folder}]}
    )
    print("Creating " + str(zip_path) + "...")
    zipdir(dvc_path, zip_path)
    print("Done!")
    file_drive.SetContentFile(zip_path)
    print("Uploading " + str(zip_path) + "...")
    file_drive.Upload()
    print("Done!")
  os.remove(zip_path)

