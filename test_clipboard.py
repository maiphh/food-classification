from io import BytesIO
import base64
from PIL import Image

from src.app import image_from_data_url

def test_image_from_data_url():
    # create a simple red image and encode it as data URL
    img = Image.new('RGB', (5, 5), color='red')
    buf = BytesIO()
    img.save(buf, format='PNG')
    data_url = 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode('utf-8')

    decoded = image_from_data_url(data_url)
    assert decoded is not None
    assert decoded.size == (5, 5)
