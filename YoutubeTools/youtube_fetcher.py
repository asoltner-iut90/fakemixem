import yt_dlp


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
        return {
            'title': info.get('title'),
            'duration': info.get('duration'),  # en secondes
            'view_count': info.get('view_count'),
            'like_count': info.get('like_count'),
            'uploader': info.get('uploader'),
            'description': info.get('description'),
            'upload_date': info.get('upload_date'),
            'thumbnail': info.get('thumbnail'),
            'formats': len(info.get('formats', [])),
        }


#video_url = "https://www.youtube.com/watch?v=1KhPXq3k45A"
#video_infos = get_video_info(video_url)

channel_url = "https://www.youtube.com/@Amixem"
video_urls = get_all_channel_videos(channel_url)
print(video_urls)


