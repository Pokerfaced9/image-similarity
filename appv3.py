import streamlit as st
import pandas as pd
import os
import requests
import atexit
import shutil
import time
from typing import Dict, List, Tuple, Optional, Set
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import imagehash
from io import BytesIO
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from functools import lru_cache
import hashlib
import json
import re
from google_auth_oauthlib.flow import Flow
import pickle

# ======================
# AUTHENTICATION CONFIG
# ======================
AUTH_FILE = "auth.json"
CREDENTIALS_FILE = "credentials.json"
TOKEN_FILE = "token.pickle"
SCOPES = ['https://www.googleapis.com/auth/userinfo.email', 'https://www.googleapis.com/auth/userinfo.profile']
ALLOWED_EMAIL_DOMAINS = ["unicommerce.com"]

def init_auth():
    """Initialize authentication file if it doesn't exist"""
    if not os.path.exists(AUTH_FILE):
        auth_data = {
            "users": {},
            "admin_emails": ["admin@unicommerce.com"]  # Add admin emails here
        }
        with open(AUTH_FILE, "w") as f:
            json.dump(auth_data, f)

def get_google_flow():
    """Create and return a Google OAuth flow"""
    return Flow.from_client_secrets_file(
        CREDENTIALS_FILE,
        scopes=SCOPES,
        redirect_uri="http://localhost:8501"  # Streamlit's default port
    )

def is_valid_work_email(email: str) -> bool:
    """Validate if the email is from allowed domains"""
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(email_pattern, email):
        return False
    
    domain = email.split('@')[1]
    return domain.lower() in ALLOWED_EMAIL_DOMAINS

def login_page():
    """Render simple login page"""
    st.title("🔐 Login")
    
    with st.form("login_form"):
        email = st.text_input("Enter your Unicommerce email")
        submit = st.form_submit_button("Login")
        
        if submit:
            if is_valid_work_email(email):
                st.session_state["authenticated"] = True
                st.session_state["email"] = email
                st.rerun()
            else:
                st.error("❌ Please use your Unicommerce email address (@unicommerce.com)")

# ======================
# CONFIGURATION
# ======================
@dataclass
class Config:
    TEMP_DIR: str = "temp_image_cache"
    IMAGE_RETENTION_MINUTES: int = 60
    HASH_THRESHOLD: int = 2
    MAX_WORKERS: int = 20
    TIMEOUT: int = 10
    CHUNK_SIZE: int = 1024
    REQUIRED_COLUMNS: Set[str] = field(default_factory=lambda: {
        "Channel Name", "Channel Product Id", "Seller SKU Code",
        "Product name", "Channel Code", "Image URL"
    })

config = Config()
os.makedirs(config.TEMP_DIR, exist_ok=True)

# ======================
# DATA MODELS
# ======================
@dataclass
class ImageMetadata:
    channel_name: str
    channel_product_id: str
    seller_sku_code: str
    product_name: str
    channel_code: str

# ======================
# UTILITY FUNCTIONS
# ======================
def cleanup_old_images() -> int:
    """Delete images older than retention period"""
    now = time.time()
    deleted_count = 0
    for filename in os.listdir(config.TEMP_DIR):
        filepath = os.path.join(config.TEMP_DIR, filename)
        if os.path.isfile(filepath):
            file_age = now - os.path.getmtime(filepath)
            if file_age > config.IMAGE_RETENTION_MINUTES * 60:
                try:
                    os.remove(filepath)
                    deleted_count += 1
                except Exception as e:
                    st.warning(f"Couldn't delete {filename}: {str(e)}")
    return deleted_count

def full_cleanup():
    """Force delete all temp files"""
    if os.path.exists(config.TEMP_DIR):
        shutil.rmtree(config.TEMP_DIR)
    os.makedirs(config.TEMP_DIR, exist_ok=True)

@lru_cache(maxsize=1000)
def download_image(url: str, img_path: str) -> bool:
    """Download and save image with caching and streaming"""
    try:
        # Stream the download in chunks
        with requests.get(url, timeout=config.TIMEOUT, stream=True) as response:
            if response.status_code == 200:
                with open(img_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=config.CHUNK_SIZE):
                        if chunk:
                            f.write(chunk)
                return True
        return False
    except Exception as e:
        return False

@lru_cache(maxsize=1000)
def calculate_hash(img_path: str) -> Optional[imagehash.ImageHash]:
    """Calculate image hash with caching"""
    try:
        with Image.open(img_path) as img:
            return imagehash.phash(img)
    except Exception as e:
        st.warning(f"Failed to calculate hash for {img_path}: {str(e)}")
        return None

# ======================
# PROCESSING FUNCTIONS
# ======================
def process_excel_file(df: pd.DataFrame, tenant: str) -> Tuple[Dict[str, List[str]], Dict[str, ImageMetadata]]:
    """Process Excel file and return grouped images and metadata"""
    image_info: Dict[str, ImageMetadata] = {}
    hashes: Dict[str, imagehash.ImageHash] = {}
    
    # Create progress containers
    progress_container = st.empty()
    download_status = st.empty()
    download_progress = st.progress(0)
    hash_status = st.empty()
    hash_progress = st.progress(0)
    group_status = st.empty()
    group_progress = st.progress(0)
    
    # Download images in parallel with batching
    download_status.write("📥 Downloading images...")
    total_images = len(df)
    downloaded_count = 0
    failed_downloads = []
    
    # Process in batches for better progress tracking
    batch_size = 100
    
    with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
        # First, create all metadata
        for idx, row in df.iterrows():
            img_name = f"{tenant + '_' if tenant else ''}{row['Channel Product Id']}_{row['Channel Code']}.jpg"
            img_path = os.path.join(config.TEMP_DIR, img_name)
            
            image_info[img_path] = ImageMetadata(
                channel_name=row['Channel Name'],
                channel_product_id=row['Channel Product Id'],
                seller_sku_code=row['Seller SKU Code'],
                product_name=row['Product name'],
                channel_code=row['Channel Code']
            )
        
        # Process downloads in batches
        for start_idx in range(0, total_images, batch_size):
            end_idx = min(start_idx + batch_size, total_images)
            batch_df = df.iloc[start_idx:end_idx]
            
            # Submit batch of downloads
            futures = []
            for idx, row in batch_df.iterrows():
                img_name = f"{tenant + '_' if tenant else ''}{row['Channel Product Id']}_{row['Channel Code']}.jpg"
                img_path = os.path.join(config.TEMP_DIR, img_name)
                
                if not os.path.exists(img_path):
                    future = executor.submit(download_image, row['Image URL'], img_path)
                    futures.append((future, img_path))
                else:
                    downloaded_count += 1
            
            # Wait for batch to complete
            for future, img_path in futures:
                try:
                    if future.result():
                        downloaded_count += 1
                    else:
                        failed_downloads.append(img_path)
                except Exception as e:
                    failed_downloads.append(img_path)
                
                # Update progress
                progress = downloaded_count / total_images
                download_progress.progress(progress)
                download_status.write(f"📥 Downloading images... ({downloaded_count}/{total_images})")
                
                # Show download speed
                if downloaded_count % 10 == 0:
                    speed = downloaded_count / (time.time() - start_time) if 'start_time' in locals() else 0
                    download_status.write(f"📥 Downloading images... ({downloaded_count}/{total_images}) - {speed:.1f} images/sec")
    
    if failed_downloads:
        st.warning(f"⚠️ Failed to download {len(failed_downloads)} images")
    
    # Calculate hashes
    hash_status.write("🔍 Calculating image hashes...")
    total_hashes = len(image_info)
    processed_hashes = 0
    
    with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
        hash_futures = []
        for img_path in image_info.keys():
            if os.path.exists(img_path):
                hash_futures.append(executor.submit(calculate_hash, img_path))
            processed_hashes += 1
            progress = processed_hashes / total_hashes
            hash_progress.progress(progress)
            hash_status.write(f"🔍 Calculating image hashes... ({processed_hashes}/{total_hashes})")
        
        # Collect hash results
        for img_path, future in zip(image_info.keys(), hash_futures):
            try:
                if img_hash := future.result():
                    hashes[img_path] = img_hash
            except Exception as e:
                st.warning(f"Failed to calculate hash for {img_path}")
    
    # Group similar images
    group_status.write("🧩 Grouping similar images...")
    group_list: List[List[str]] = []
    visited: Set[str] = set()
    total_to_process = len(hashes)
    processed_groups = 0
    
    for idx, (img1, hash1) in enumerate(hashes.items()):
        if img1 in visited:
            continue
            
        group = [img2 for img2, hash2 in hashes.items() 
                if img2 not in visited and hash1 - hash2 <= config.HASH_THRESHOLD]
        
        if len(group) > 1:
            group_list.append(group)
        
        visited.update(group)
        
        # Update grouping progress
        processed_groups += 1
        progress = processed_groups / total_to_process
        group_progress.progress(progress)
        group_status.write(f"🧩 Grouping similar images... ({processed_groups}/{total_to_process})")
    
    # Show final summary
    progress_container.write(f"""
    ### ✅ Processing Complete
    - 📥 Downloaded: {downloaded_count} images
    - 🔍 Processed: {len(hashes)} hashes
    - 🧩 Created: {len(group_list)} groups
    """)
    
    # Clear progress indicators after a short delay
    time.sleep(1)  # Give users time to see the final progress
    download_status.empty()
    download_progress.empty()
    hash_status.empty()
    hash_progress.empty()
    group_status.empty()
    group_progress.empty()
    
    return {i: g for i, g in enumerate(group_list)}, image_info

# ======================
# UI COMPONENTS
# ======================
def render_image_group(group_id: int, images: List[str], image_info: Dict[str, ImageMetadata]):
    """Render a group of images with their metadata"""
    # Clean up missing images first
    cleanup_missing_images()
    
    # Skip rendering if no valid images
    if not images:
        st.info("No valid images in this group")
        return
        
    st.markdown(f"### Group {group_id + 1} ({len(images)} images)")
    
    # Create a container for the image grid
    image_container = st.container()
    
    with image_container:
        cols = st.columns(min(5, len(images)))
        for idx, img_path in enumerate(images):
            with cols[idx % 5]:
                meta = image_info.get(img_path)
                if not meta:
                    continue
                    
                # Get base64 image data
                base64_img = get_base64_image(img_path)
                if not base64_img:
                    continue
                    
                # Create unique keys for each element using group_id and idx
                img_key = f"img_{group_id}_{idx}"
                sku_key = f"sku_{group_id}_{idx}"
                remove_key = f"remove_{group_id}_{idx}"
                
                # Card front: image + product name only
                st.markdown(
                    f"""
                    <div style="
                        display: flex;
                        flex-direction: column;
                        justify-content: flex-start;
                        align-items: stretch;
                        height: 460px;
                        background: #232323;
                        border-radius: 2px;
                        padding: 8px 8px 4px 8px;
                        margin-bottom: 6px;
                        box-shadow: 0 1px 4px rgba(0,0,0,0.07);
                    ">
                        <div style='height: 220px; display: flex; align-items: center; justify-content: center; overflow: hidden; border-radius: 6px;'>
                            <img src="data:image/jpeg;base64,{base64_img}" style="max-height: 100%; max-width: 100%; object-fit: contain;">
                        </div>
                        <div style="
                            flex: 1 1 auto;
                            padding: 12px 0;
                            color: #fff;
                            overflow-y: auto;
                        ">
                            <div style="
                                font-weight: bold;
                                font-size: 16px;
                                line-height: 1.4;
                                margin-bottom: 12px;
                                word-wrap: break-word;
                            ">
                                {meta.product_name}
                            </div>
                            <div style="font-size: 14px; color: #ccc; line-height: 1.6;">
                                <div style="margin-bottom: 4px; word-wrap: break-word;"><b>Channel ID:</b> {meta.channel_product_id}</div>
                                <div style="margin-bottom: 4px; word-wrap: break-word;"><b>SKU:</b> {meta.seller_sku_code}</div>
                                <div style="margin-bottom: 4px; word-wrap: break-word;"><b>Channel:</b> {meta.channel_code}</div>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Style the input and button
                st.markdown("""
                <style>
                .stTextInput input {
                    font-size: 16px !important;
                    padding: 8px 12px !important;
                }
                .stButton button {
                    font-size: 16px !important;
                    padding: 4px 12px !important;
                }
                </style>
                """, unsafe_allow_html=True)
                
                sku = st.text_input(
                    f"Uniware SKU",
                    key=sku_key,
                    value=st.session_state.image_level_sku_codes.get(img_path, "")
                )
                if st.button("Remove", key=remove_key):
                    # Remove from the group in session state
                    current_group = st.session_state.ordered_groups[group_id]
                    current_group.remove(img_path)
                    st.session_state.ordered_groups[group_id] = current_group
                    
                    # Add to unassigned images
                    if img_path not in st.session_state.unassigned_images:
                        st.session_state.unassigned_images.append(img_path)
                    
                    # If group becomes empty or has only one image, move remaining to unassigned
                    if len(current_group) <= 1:
                        for remaining_img in current_group:
                            if remaining_img not in st.session_state.unassigned_images:
                                st.session_state.unassigned_images.append(remaining_img)
                        del st.session_state.ordered_groups[group_id]
                    
                    st.rerun()

def get_base64_image(img_path: str) -> str:
    """Convert image to base64 for HTML display"""
    import base64
    try:
        if not os.path.exists(img_path):
            return ""
        with open(img_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    except Exception as e:
        st.warning(f"Failed to load image {img_path}: {str(e)}")
        return ""

def cleanup_missing_images():
    """Remove non-existent images from groups and unassigned images"""
    # Clean up ordered groups
    for group_id in list(st.session_state.ordered_groups.keys()):
        valid_images = [img for img in st.session_state.ordered_groups[group_id] if os.path.exists(img)]
        if len(valid_images) == 0:
            del st.session_state.ordered_groups[group_id]
        else:
            st.session_state.ordered_groups[group_id] = valid_images
    
    # Clean up unassigned images
    st.session_state.unassigned_images = [img for img in st.session_state.unassigned_images if os.path.exists(img)]

def render_unassigned_images():
    """Render unassigned images with consistent height and compact UI"""
    if st.session_state.unassigned_images:
        cols = st.columns(5)
        for idx, img_path in enumerate(st.session_state.unassigned_images):
            with cols[idx % 5]:
                meta = st.session_state.image_info.get(img_path)
                # Create unique keys for unassigned images
                unassigned_key = f"unassigned_{idx}"
                sku_key = f"unassigned_sku_{idx}"
                reassign_key = f"reassign_{idx}"
                reassign_group_key = f"reassign_group_{idx}"
                
                st.markdown(
                    f"""
                    <div style="
                        display: flex;
                        flex-direction: column;
                        justify-content: flex-start;
                        align-items: stretch;
                        height: 460px;
                        background: #232323;
                        border-radius: 8px;
                        padding: 8px 8px 4px 8px;
                        margin-bottom: 6px;
                        box-shadow: 0 1px 4px rgba(0,0,0,0.07);
                    ">
                        <div style='height: 220px; display: flex; align-items: center; justify-content: center; overflow: hidden; border-radius: 6px;'>
                            <img src="data:image/jpeg;base64,{get_base64_image(img_path)}" style="max-height: 100%; max-width: 100%; object-fit: contain;">
                        </div>
                        <div style="
                            flex: 1 1 auto;
                            padding: 12px 0;
                            color: #fff;
                            overflow-y: auto;
                        ">
                            <div style="
                                font-weight: bold;
                                font-size: 16px;
                                line-height: 1.4;
                                margin-bottom: 12px;
                                word-wrap: break-word;
                            ">
                                {meta.product_name}
                            </div>
                            <div style="font-size: 14px; color: #ccc; line-height: 1.6;">
                                <div style="margin-bottom: 4px; word-wrap: break-word;"><b>Channel ID:</b> {meta.channel_product_id}</div>
                                <div style="margin-bottom: 4px; word-wrap: break-word;"><b>SKU:</b> {meta.seller_sku_code}</div>
                                <div style="margin-bottom: 4px; word-wrap: break-word;"><b>Channel:</b> {meta.channel_code}</div>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                with st.expander("Show details", expanded=False):
                    st.markdown(f"""
                        <span style='font-size: 12px;'>
                        <b>Product:</b> {meta.product_name}<br>
                        <b>Channel ID:</b> {meta.channel_product_id}<br>
                        <b>SKU:</b> {meta.seller_sku_code}<br>
                        <b>Channel:</b> {meta.channel_code}
                        </span>
                    """, unsafe_allow_html=True)
                sku = st.text_input(
                    f"Assign SKU",
                    key=sku_key,
                    value=st.session_state.image_level_sku_codes.get(img_path, "")
                )
                
                # Add group reassignment controls
                st.markdown("---")
                st.markdown("**Reassign to Group**")
                
                # Create a list of available groups
                available_groups = list(st.session_state.ordered_groups.keys())
                group_names = [f"Group {g+1}" for g in available_groups]
                
                # Add option to create new group
                group_names.append("Create New Group")
                
                selected_group = st.selectbox(
                    "Select Target Group",
                    options=group_names,
                    key=reassign_group_key
                )
                
                if st.button("Reassign", key=reassign_key):
                    if selected_group == "Create New Group":
                        # Create a new group
                        new_group_id = max(available_groups) + 1 if available_groups else 0
                        st.session_state.ordered_groups[new_group_id] = [img_path]
                    else:
                        # Get the selected group ID
                        selected_group_id = available_groups[group_names.index(selected_group)]
                        # Add image to the selected group
                        st.session_state.ordered_groups[selected_group_id].append(img_path)
                    
                    # Remove from unassigned images
                    st.session_state.unassigned_images.remove(img_path)
                    st.rerun()

# ======================
# MAIN APP
# ======================
def main():
    st.set_page_config(layout="wide")
    
    # Check if user is authenticated
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
    
    if not st.session_state["authenticated"]:
        login_page()
        return
    
    st.title("Image Grouping Application")
    
    # Add custom CSS and JavaScript for scroll position and compact buttons
    st.markdown("""
    <style>
    div[data-testid="stButton"] button {
        padding: 0.25rem 0.5rem !important;
        min-width: 100px !important;
    }
    div[data-testid="stButton"] {
        margin: 0 2px !important;
    }
    </style>
    <script>
        // Store scroll position before page reload
        window.onbeforeunload = function() {
            localStorage.setItem('scrollPosition', window.scrollY);
        };
        
        // Restore scroll position after page load
        window.onload = function() {
            const scrollPosition = localStorage.getItem('scrollPosition');
            if (scrollPosition) {
                window.scrollTo(0, scrollPosition);
                localStorage.removeItem('scrollPosition');
            }
        };
    </script>
    """, unsafe_allow_html=True)
    
    # Add control buttons in the top right
    col1, col2, col3, col4 = st.columns([6, 1, 1, 1])
    with col2:
        if st.button("Clear Cache", use_container_width=True, key="clear_cache_btn"):
            deleted_count = cleanup_old_images()
            if deleted_count > 0:
                st.success(f"Cleared {deleted_count} temporary files!")
            else:
                st.info("No temporary files to clear.")
            st.rerun()
    with col3:
        if st.button("New Excel Entry", use_container_width=True, key="new_excel_btn"):
            # Clear all session state except authentication
            auth_state = st.session_state.get("authenticated", False)
            email = st.session_state.get("email", "")
            
            # Clear all session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            
            # Restore authentication state
            st.session_state["authenticated"] = auth_state
            st.session_state["email"] = email
            
            # Clear temp directory
            full_cleanup()
            
            st.success("Ready for new Excel upload!")
            st.rerun()
    with col4:
        if st.button("Logout", use_container_width=True, key="logout_btn"):
            if os.path.exists(TOKEN_FILE):
                os.remove(TOKEN_FILE)
            st.session_state["authenticated"] = False
            st.rerun()
    
    # Initialize session state
    session_defaults = {
        "ordered_groups": {},
        "ungrouped_images": [],
        "image_info": {},
        "removed_images": [],
        "uniware_sku_codes": {},
        "image_level_sku_codes": {},
        "unassigned_images": [],
        "show_unassigned": False,
        "current_group_idx": 0,
        "grouped": False,
        "last_cleanup": datetime.now(),
        "tenant_name": ""  # Add tenant name to session state
    }
    
    for key, val in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val
    
    # Automatic cleanup
    if (datetime.now() - st.session_state.last_cleanup) > timedelta(minutes=5):
        deleted = cleanup_old_images()
        if deleted > 0:
            st.toast(f"Auto-cleaned {deleted} old images", icon="🧹")
            # Clean up missing images after cleanup
            cleanup_missing_images()
        st.session_state.last_cleanup = datetime.now()
    
    # Only show input section if not grouped
    if not st.session_state.grouped:
        input_container = st.container()
        with input_container:
            # Main content
            tenant = st.text_input("Enter Tenant Name", key="tenant_input")
            st.session_state.tenant_name = tenant  # Store tenant name in session state
            
            # Only show file uploader if tenant name is provided
            if not tenant:
                st.warning("⚠️ Please enter a tenant name before uploading the file.")
                uploaded_file = None
            else:
                uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])
            
            if uploaded_file:
                try:
                    df = pd.read_excel(uploaded_file)
                    if not config.REQUIRED_COLUMNS.issubset(df.columns):
                        st.error("❌ Excel must contain the required columns.")
                    else:
                        with st.status("🚀 Processing dataset...", expanded=True) as status:
                            ordered_groups, image_info = process_excel_file(df, tenant)
                            st.session_state.ordered_groups = ordered_groups
                            st.session_state.image_info = image_info
                            
                            # Initialize SKU codes
                            for group_id, img_list in st.session_state.ordered_groups.items():
                                st.session_state.uniware_sku_codes[group_id] = ""
                                for img_path in img_list:
                                    st.session_state.image_level_sku_codes[img_path] = ""
                            
                            st.session_state.grouped = True
                            status.update(label="✅ Processing complete!", state="complete")
                            st.rerun()
                except Exception as e:
                    st.error(f"❌ Processing failed: {str(e)}")
                    st.session_state.grouped = False
    
    if st.session_state.grouped:
        total_groups = list(st.session_state.ordered_groups.keys())
        
        if total_groups:
            # Ensure current_group_idx is valid
            if st.session_state.current_group_idx >= len(total_groups):
                st.session_state.current_group_idx = 0
            
            current_group = total_groups[st.session_state.current_group_idx]
            imgs = st.session_state.ordered_groups[current_group]

            if len(imgs) == 0:
                # If current group is empty, move to next group
                st.session_state.current_group_idx = (st.session_state.current_group_idx + 1) % len(total_groups)
                st.rerun()

            # Add navigation controls at the top
            st.markdown("""
            <style>
            .nav-container {
                margin-bottom: 1rem;
            }
            .navbar {
                display: flex;
                align-items: center;
                gap: 8px;
                background: #18191A;
                padding: 0.75rem;
                border-radius: 8px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }
            .nav-group {
                display: flex;
                align-items: center;
                gap: 8px;
            }
            .nav-group.buttons {
                flex: 0 0 auto;
            }
            .nav-group.input {
                flex: 1;
                margin-left: 8px;
            }
            .nav-button {
                background: #232323;
                border: 1px solid #404040;
                color: white;
                padding: 0.5rem 1rem;
                border-radius: 6px;
                cursor: pointer;
                font-size: 14px;
                min-height: 38px;
                min-width: 90px;
                transition: all 0.2s;
            }
            .nav-button:hover {
                background: #2A2A2A;
                border-color: #505050;
            }
            .nav-input {
                width: 100%;
                background: #232323;
                border: 1px solid #404040;
                color: white;
                padding: 0.5rem 1rem;
                border-radius: 6px;
                font-size: 14px;
                min-height: 38px; 
            }
            /* Override Streamlit's default styles */
            .nav-container [data-testid="stHorizontalBlock"] {
                gap: 0 !important;
            }
            .nav-container .stTextInput > div {
                margin-bottom: 0 !important;
            }
            .nav-container .stTextInput > label {
                display: none !important;
            }
            .nav-container [data-testid="stButton"] {
                margin: 0 !important;
            }
            .nav-container [data-testid="stButton"] button {
                border: 1px solid #404040 !important;
                background-color: #232323 !important;
                border-radius: 6px !important;
                padding: 0.5rem 0.75rem !important;
                min-height: 38px !important;
                font-size: 14px !important;
            }
            .nav-container [data-testid="stButton"] button:hover {
                border-color: #505050 !important;
                background-color: #2A2A2A !important;
            }
            .nav-container .stTextInput > div > div > input {
                background: #232323 !important;
                border: 1px solid #404040 !important;
                color: #fff !important;
                padding: 0.5rem 1rem !important;
                min-height: 38px !important;
                border-radius: 6px !important;
                font-size: 14px !important;
            }
            </style>
            """, unsafe_allow_html=True)

            # Create the navbar container
            st.markdown('<div class="nav-container"><div class="navbar">', unsafe_allow_html=True)
            
            # Navigation buttons
            col1, col2 = st.columns([4, 3])
            with col1:
                subcol1, subcol2, subcol3, subcol4 = st.columns([1,1,1,1])
                with subcol1:
                    if st.button("⏮ First", use_container_width=True, key=f"first_{current_group}"):
                        st.session_state.current_group_idx = 0
                        st.rerun()
                with subcol2:
                    if st.button("◀ Previous", use_container_width=True, key=f"prev_{current_group}"):
                        st.session_state.current_group_idx = (st.session_state.current_group_idx - 1) % len(st.session_state.ordered_groups)
                        st.rerun()
                with subcol3:
                    if st.button("▶ Next", use_container_width=True, key=f"next_{current_group}"):
                        st.session_state.current_group_idx = (st.session_state.current_group_idx + 1) % len(st.session_state.ordered_groups)
                        st.rerun()
                with subcol4:
                    if st.button("⏭ Last", use_container_width=True, key=f"last_{current_group}"):
                        st.session_state.current_group_idx = len(st.session_state.ordered_groups) - 1
                        st.rerun()
            
            # SKU input
            with col2:
                group_sku = st.text_input(
                    "",  # Empty label
                    key=f"group_sku_{current_group}",
                    value=st.session_state.uniware_sku_codes.get(current_group, ""),
                    placeholder=f"Uniware SKU for Group {current_group + 1}"
                )
                if group_sku:
                    st.session_state.uniware_sku_codes[current_group] = group_sku

            st.markdown('</div></div>', unsafe_allow_html=True)

            render_image_group(current_group, imgs, st.session_state.image_info)
            
            # Unassigned images section
            show_unassigned = st.toggle("Show Unassigned Images", value=st.session_state.show_unassigned, key="show_unassigned_main")
            if show_unassigned:
                st.markdown("---")
                st.subheader("Unassigned Images")
                if st.session_state.unassigned_images:
                    render_unassigned_images()
                else:
                    st.info("No unassigned images")
        else:
            st.info("No groups available. Please upload a file to start grouping images.")

if __name__ == "__main__":
    main()
    atexit.register(full_cleanup)