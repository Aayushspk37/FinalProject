# Diagnosis/urls.py
from django.contrib import admin
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from . import views

app_name = 'Diagnosis'

urlpatterns = [
    # ============================================
    # MAIN PAGES (Public)
    # ============================================
    path('', views.index, name="index"),
    path('index/', views.index, name="index"),
    path('about/', views.about, name="about"),
    path('services/', views.services, name="services"),
    path('contact/', views.contact, name="contact"),
    path('faq/', views.faq, name="faq"),
    path('modalities/', views.modalities, name="modalities"),
    path('pricing/', views.pricing, name="pricing"),
    path('privacy/', views.privacy, name="privacy"),
    path('support/', views.support, name="support"),
    path('team/', views.team, name="team"),
    path('terms/', views.terms, name="terms"),
    path('documentation/', views.documentation, name="documentation"),
    
    # ============================================
    # AUTHENTICATION
    # ============================================
    path('login/', views.user_login, name="login"),
    path('register/', views.register, name="register"),
    path('logout/', views.user_logout, name="logout"),
    
    # ============================================
    # PROTECTED PAGES (Login Required)
    # ============================================
    path('dashboard/', views.dashboard_view, name="dashboard"),
    path('result/', views.result, name="result"),
    path('report/', views.report, name="report"),
    path('history/', views.history_page, name='history'),
    path('diagnose/', views.diagnose_page, name='diagnose'),
    
    # ============================================
    # REPORTS
    # ============================================
    path('reports/', views.reports_view, name='reports'),
    path('reports/<int:report_id>/', views.report_detail_view, name='report_detail'),
    path('reports/<int:report_id>/download/', views.download_report_view, name='download_report'),
    
    # ============================================
    # PROFILE
    # ============================================
    path('profile/', views.profile_view, name='profile'),
    
    # ============================================
    # NOTIFICATIONS
    # ============================================
    path('notifications/', views.notifications_view, name='notifications'),
    path('notifications/mark-read/<int:notification_id>/', views.mark_notification_read, name='mark_notification_read'),
    path('notifications/delete/<int:notification_id>/', views.delete_notification, name='delete_notification'),
    
    # ============================================
    # FEEDBACK
    # ============================================
    path('feedback/', views.feedback_view, name='feedback'),
    path('feedback/submit/', views.submit_feedback, name='submit_feedback'),
    path('feedback/delete/<int:feedback_id>/', views.delete_feedback, name='delete_feedback'),
    
    # ============================================
    # API ENDPOINTS (REST API)
    # ============================================
    # Diagnosis History API
    path('api/diagnosis/history/', views.get_diagnosis_history, name='diagnosis_history'),
    path('api/diagnosis/<int:record_id>/delete/', views.delete_diagnosis, name='delete_diagnosis'),
    path('api/diagnosis/clear-history/', views.clear_history, name='clear_history'),
    path('api/diagnosis/<int:record_id>/detail/', views.get_record_detail, name='record_detail'),
    
    # Diagnosis API - Model Integration
    path('api/diagnose/', views.diagnose_image, name='diagnose_image'),
    path('api/diagnosis/save-history/', views.save_history, name='save_history'),
    
    # AI Interpretation API
    path('api/generate-interpretation/', views.generate_interpretation, name='generate_interpretation'),
    
    # ============================================
    # TEST/DEVELOPMENT ENDPOINTS (Superuser only)
    # ============================================
    path('api/test/', views.test_api, name='test_api'),
    path('api/test-models/', views.test_model_loading, name='test_model_loading'),
    path('api/diagnosis/create-test/', views.create_test_records, name='create_test_records'),
    
    # ============================================
    # REDIRECTS
    # ============================================
    path('home/', views.home, name='home'),
    
    path('test-api-key/', views.test_api_key_status, name='test_api_key'),
    
    path('profile/upload-avatar/', views.upload_avatar, name='upload_avatar'),
    
    path('change-password/', views.change_password, name='change_password'),
    
    path('api/password-reset/', views.password_reset, name='password_reset'),
]

# ============================================
# MEDIA AND STATIC FILES (Development)
# ============================================
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)