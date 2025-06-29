from beam import endpoint, Image, Volume, env, function

IMAGE = Image(
    python_version="python3.11",
    python_packages=[
        "tqdm",
    ]
)

CHATTERBOX_PROJECT = "./chatterbox-project"

def extract_single_tar(args):
    tar_path, extract_base_dir, lang_dir = args
    import os
    import tarfile
    import logging
    
    logger = logging.getLogger(__name__)
    tar_filename = os.path.basename(tar_path)
    tar_name = os.path.splitext(tar_filename)[0]
    
    tar_extract_dir = os.path.join(extract_base_dir, tar_name)
    os.makedirs(tar_extract_dir, exist_ok=True)
    
    try:
        with tarfile.open(tar_path, 'r') as tar:
            tar.extractall(path=tar_extract_dir)
        
        return {
            "success": True,
            "file": tar_filename,
            "lang": lang_dir,
            "extract_dir": tar_extract_dir
        }
    except Exception as e:
        return {
            "success": False,
            "file": tar_filename,
            "lang": lang_dir,
            "error": str(e)
        }

@function(
    image=IMAGE,
    memory=32,
    cpu=4,
    # gpu="T4",
    volumes=[Volume(name="chatterbox-project", mount_path=CHATTERBOX_PROJECT)],
    timeout=-1
)
def extract_emilia(langauge_dirs):
    import os
    import logging
    from multiprocessing import Pool, cpu_count
    from tqdm import tqdm
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    emilia_base_path = os.path.join(CHATTERBOX_PROJECT, "Emilia-YODAS")
    
    if not os.path.exists(emilia_base_path):
        logger.error(f"Emilia-YODAS directory not found at {emilia_base_path}")
        return {"error": "Emilia-YODAS directory not found"}
    
    all_tasks = []
    extraction_results = {}
    
    for lang_dir in language_dirs:
        lang_path = os.path.join(emilia_base_path, lang_dir)
        
        if not os.path.exists(lang_path):
            logger.warning(f"Language directory {lang_dir} not found, skipping")
            continue
            
        tar_files = [f for f in os.listdir(lang_path) if f.endswith('.tar')]
        tar_files.sort()
        
        logger.info(f"Found {len(tar_files)} tar files in {lang_dir}")
        
        extract_base_dir = os.path.join(lang_path, f"{lang_dir}")
        os.makedirs(extract_base_dir, exist_ok=True)
        
        extraction_results[lang_dir] = {
            "total_files": len(tar_files),
            "extracted": 0,
            "failed": 0,
            "errors": []
        }
        
        for tar_file in tar_files:
            tar_path = os.path.join(lang_path, tar_file)
            all_tasks.append((tar_path, extract_base_dir, lang_dir))
    
    num_processes = cpu_count()
    logger.info(f"Starting extraction with {num_processes} processes for {len(all_tasks)} files")
    
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap(extract_single_tar, all_tasks),
            total=len(all_tasks),
            desc="Extracting files"
        ))
    
    total_extracted = 0
    for result in results:
        lang = result["lang"]
        if result["success"]:
            extraction_results[lang]["extracted"] += 1
            total_extracted += 1
            logger.info(f"Successfully extracted {result['file']} to {result['extract_dir']}")
        else:
            extraction_results[lang]["failed"] += 1
            error_msg = f"Failed to extract {result['file']}: {result['error']}"
            extraction_results[lang]["errors"].append(error_msg)
            logger.error(error_msg)
    
    logger.info(f"Extraction complete. Total files extracted: {total_extracted}")
    
    return {
        "success": True,
        "total_extracted": total_extracted,
        "total_processes": num_processes,
        "results": extraction_results
    }

if __name__ == "__main__":
    language_dirs = ["DE", "FR", "EN"]
    choice = input("1: Local\n2: Remote")
    if choice == "1":
        extract_emilia.local(language_dirs)
    elif choice == "2":
        extract_emilia.remote(language_dirs)
    