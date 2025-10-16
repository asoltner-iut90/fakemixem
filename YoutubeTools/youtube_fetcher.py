import yt_dlp
import requests

def get_dislikes(video_id):
    """
    Récupère les dislikes via l'API Return YouTube Dislike.

    Args:
        video_id (str): ID de la vidéo YouTube

    Returns:
        dict: Statistiques incluant les dislikes estimés
    """
    try:
        url = f"https://returnyoutubedislikeapi.com/votes?videoId={video_id}"
        response = requests.get(url, timeout=5)

        if response.status_code == 200:
            data = response.json()
            return {
                'likes': data.get('likes', 0),
                'dislikes': data.get('dislikes', 0),
                'rating': data.get('rating', 0),
                'viewCount': data.get('viewCount', 0),
            }
    except Exception as e:
        print(f"Erreur lors de la récupération des dislikes: {e}")

    return None

def get_all_channel_videos(channel_url):
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'extract_flat': True,
        'skip_download': True,
    }

    video_urls = []

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        # Ajouter /videos pour obtenir tous les uploads
        if not channel_url.endswith('/videos'):
            channel_url = channel_url.rstrip('/') + '/videos'

        print(f"Récupération des vidéos de la chaîne...")
        info = ydl.extract_info(channel_url, download=False)

        if 'entries' in info:
            for entry in info['entries']:
                if entry and 'id' in entry:
                    video_id = entry['id']
                    video_urls.append(f"https://www.youtube.com/watch?v={video_id}")

        print(f"✓ {len(video_urls)} vidéos trouvées")

    return video_urls

def get_video_info(url):
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'skip_download': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)

        # Informations disponibles
        return info




