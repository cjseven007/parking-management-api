import firebase_admin
from firebase_admin import credentials, firestore
from app.core.config import settings


def init_firebase():
    if firebase_admin._apps:
        return firebase_admin.get_app()

    if settings.firebase_credentials:
        cred = credentials.Certificate(settings.firebase_credentials)
        return firebase_admin.initialize_app(cred)

    # Cloud Run / Google environment
    cred = credentials.ApplicationDefault()
    return firebase_admin.initialize_app(cred)


init_firebase()
db = firestore.client()