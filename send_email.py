import requests
import os
from github import Github
from git import Repo

# GitHub configuration
GITHUB_TOKEN = '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'
g = Github(GITHUB_TOKEN)
REPO_NAME = g.get_repo('beckerman1101/daily_accum_mapping')  # Replace with your repository
BRANCH_NAME = 'main'  # Adjust if needed
FILE_PATH_IN_REPO = 'daily_file/filename.tar.gz'  # Path where the file will be saved in your repo
DOWNLOAD_URL = 'https://tgftp.nws.noaa.gov/SL.us008001/DF.sha/DC.cap/DS.WWA/current_all.tar.gz'

# Download the file from the URL
def download_file(url, filename):
    response = requests.get(url)
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"File downloaded successfully: {filename}")
    else:
        print(f"Failed to download file: {response.status_code}")

# Push the file to GitHub repository
def push_to_github(file_path, repo_name, branch_name, github_token):
    # Authenticate with GitHub
    g = Github(github_token)
    repo = g.get_repo(repo_name)
    
    # Ensure we're working with the correct branch
    try:
        branch_ref = repo.get_git_ref(f"heads/{branch_name}")
    except:
        print(f"Error: Branch {branch_name} does not exist.")
        return

    # Upload the file to the repository
    with open(file_path, 'rb') as file_content:
        repo.create_file(FILE_PATH_IN_REPO, "Add downloaded file", file_content.read(), branch=branch_name)

    print(f"File successfully pushed to GitHub at {repo_name}/{branch_name}")

# Main function
def main():
    # Download the file
    downloaded_filename = 'downloaded_file.tar.gz'
    download_file(DOWNLOAD_URL, downloaded_filename)
    
    # Push the downloaded file to GitHub repository
    push_to_github(downloaded_filename, REPO_NAME, BRANCH_NAME, GITHUB_TOKEN)

    # Optionally, you can clean up by deleting the downloaded file
    os.remove(downloaded_filename)

if __name__ == "__main__":
    main()
