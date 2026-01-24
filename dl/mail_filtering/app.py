import os
import sys
import time
import requests
import torch
from dotenv import load_dotenv
from msal import PublicClientApplication, SerializableTokenCache
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ==========================================
#              CONFIGURATION
# ==========================================
# 1. Azure Setup
load_dotenv()
CLIENT_ID = os.getenv("CLIENT_ID")
TENANT_ID = "common"
SCOPES = ["Mail.ReadWrite"]
CACHE_FILE = "token_cache.bin"

# 2. Model Setup
MODEL_PATH = "mardonbekhazratov/bert-base-uncased-fine-tuned"
LABELS = ['happy', 'not-relevant', 'angry', 'disgust', 'sad', 'surprise']
CONFIDENCE_THRESHOLD = 0.5  # Only move if model is 50% sure

# 3. Folder Setup
ROOT_FOLDER_NAME = "filter_by_mood"

# 4. Hardware Setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
#           PART 1: AUTHENTICATION
# ==========================================
def get_access_token():
    """
    Authenticates using a persistent cache file.
    Only asks for login if the cache is missing or expired.
    """
    # 1. Load existing cache if available
    cache = SerializableTokenCache()
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            cache.deserialize(f.read())

    # 2. Setup the App with this cache
    app = PublicClientApplication(
        CLIENT_ID, 
        authority=f"https://login.microsoftonline.com/{TENANT_ID}",
        token_cache=cache
    )

    # 3. Try to get token SILENTLY (from cache) first
    accounts = app.get_accounts()
    if accounts:
        result = app.acquire_token_silent(SCOPES, account=accounts[0])
        if result:
            print(">>> ✅ Loaded login from cache (No browser needed).")
            return result['access_token']

    # 4. If Silent fails (First run or expired), do Interactive Login
    print(">>> ⚠️  Cache empty or expired. ONE-TIME LOGIN REQUIRED.")
    flow = app.initiate_device_flow(scopes=SCOPES)
    if "user_code" not in flow:
        print("Error: Could not initiate device flow.")
        sys.exit(1)
        
    print(f"\n>>> ACTION REQUIRED: Login at {flow['verification_uri']} with code: {flow['user_code']}")
    result = app.acquire_token_by_device_flow(flow)
    
    if "access_token" in result:
        print(">>> Authentication successful.\n")
        
        # 5. SAVE the cache to disk for next time
        if cache.has_state_changed:
            with open(CACHE_FILE, "w") as f:
                f.write(cache.serialize())
                
        return result['access_token']
    else:
        print(f"Authentication failed: {result.get('error_description')}")
        sys.exit(1)

# ==========================================
#        PART 2: OUTLOOK API HELPERS
# ==========================================
def get_headers(token):
    return {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}

def find_folder_id(token, folder_name, parent_id=None):
    """Recursively finds a folder ID by its name."""
    url = f"https://graph.microsoft.com/v1.0/me/mailFolders/{parent_id}/childFolders" if parent_id else \
          "https://graph.microsoft.com/v1.0/me/mailFolders?includeHiddenFolders=false&$top=50"
    
    response = requests.get(url, headers=get_headers(token))
    if response.status_code != 200: return None
    
    for folder in response.json().get('value', []):
        if folder['displayName'].lower() == folder_name.lower():
            return folder['id']
        if folder.get('childFolderCount', 0) > 0:
            found = find_folder_id(token, folder_name, folder['id'])
            if found: return found
    return None

def build_dynamic_map(token):
    """
    Automatically finds the IDs for 'happy', 'sad', etc. inside 'filter_by_mood'.
    Returns a dictionary: {0: 'ID_FOR_HAPPY', 1: 'ID_FOR_NOT_RELEVANT', ...}
    """
    print(f"--- Building Folder Map ---")
    
    # 1. Find the Root Folder
    root_id = find_folder_id(token, ROOT_FOLDER_NAME)
    if not root_id:
        print(f"CRITICAL ERROR: Could not find folder '{ROOT_FOLDER_NAME}' in your mailbox.")
        sys.exit(1)
    
    # 2. Find Subfolders matching LABELS
    folder_map = {}
    
    # Get all children of the root folder
    url = f"https://graph.microsoft.com/v1.0/me/mailFolders/{root_id}/childFolders"
    response = requests.get(url, headers=get_headers(token))
    subfolders = response.json().get('value', [])
    
    for i, label in enumerate(LABELS):
        # Find the subfolder that matches the label name
        match = next((f for f in subfolders if f['displayName'].lower() == label.lower()), None)
        if match:
            folder_map[i] = match['id']
            print(f"✅ Mapped Label '{label}' -> Folder ID: ...{match['id'][-10:]}")
        else:
            print(f"⚠️  Warning: No subfolder found for label '{label}'. Emails with this mood will remain in Inbox.")
            folder_map[i] = None
            
    return folder_map

def fetch_unread_emails(token):
    url = "https://graph.microsoft.com/v1.0/me/mailFolders/inbox/messages"
    params = {
        '$filter': 'isRead eq false',
        '$select': 'id,subject,bodyPreview',
        '$top': 10
    }
    response = requests.get(url, headers=get_headers(token), params=params)
    return response.json().get('value', [])

def move_email(token, message_id, folder_id):
    url = f"https://graph.microsoft.com/v1.0/me/messages/{message_id}/move"
    payload = {"destinationId": folder_id}
    requests.post(url, headers=get_headers(token), json=payload)

# ==========================================
#        PART 3: THE AI BRAIN (RTX 4060)
# ==========================================
def load_model():
    print(f"\n--- Loading Model on {DEVICE} ---")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        model.to(DEVICE)
        model.eval()
        print("✅ Model loaded successfully.\n")
        return tokenizer, model
    except Exception as e:
        print(f"Error loading model from {MODEL_PATH}: {e}")
        sys.exit(1)

def predict_mood(text, tokenizer, model):
    """Returns the index of the predicted label."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        confidence, predicted_class = torch.max(probs, dim=1)
        
    return predicted_class.item(), confidence.item()

# ==========================================
#           PART 4: MAIN EXECUTION
# ==========================================
def main():
    # 1. Setup
    token = get_access_token()
    tokenizer, model = load_model()
    
    # 2. Build Map
    folder_map = build_dynamic_map(token)
    
    print("\n--- Starting Filter Loop (Press Ctrl+C to stop) ---")
    
    while True:
        try:
            # 3. Fetch
            emails = fetch_unread_emails(token)
            
            if not emails:
                print("No unread emails. Waiting 60s...", end='\r')
                time.sleep(60)
                continue
                
            print(f"\nProcessing {len(emails)} new emails...")
            
            # 4. Infer & Act
            for email in emails:
                # Combine subject and preview for context
                text = f"{email['subject']} {email['bodyPreview']}"
                
                # Predict
                label_idx, confidence = predict_mood(text, tokenizer, model)
                predicted_label = LABELS[label_idx]
                
                print(f"📧 '{email['subject'][:30]}...' -> Predicted: [{predicted_label}] ({confidence:.2f})")
                
                # Move
                target_folder_id = folder_map.get(label_idx)
                
                if target_folder_id and confidence >= CONFIDENCE_THRESHOLD:
                    move_email(token, email['id'], target_folder_id)
                    print(f"   -> Moved to folder '{predicted_label}'")
                elif not target_folder_id:
                    print(f"   -> Skipped (No matching folder for '{predicted_label}')")
                else:
                    print(f"   -> Skipped (Low confidence)")
            
            # Wait before next batch
            time.sleep(10)
            
        except KeyboardInterrupt:
            print("\nStopping script.")
            break
        except Exception as e:
            print(f"Error in loop: {e}")
            time.sleep(60)

if __name__ == "__main__":
    main()