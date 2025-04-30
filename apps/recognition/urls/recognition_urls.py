from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from apps.recognition.views.recognition_view import upload_and_recognize

urlpatterns = [
    path('upload/', upload_and_recognize, name='upload_and_recognize'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)