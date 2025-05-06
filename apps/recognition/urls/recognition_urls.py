from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from apps.recognition.views.recognition_view import upload_and_recognize, register_person

urlpatterns = [
    path('upload/', upload_and_recognize, name='upload'),
    path('register/', register_person, name='register'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
