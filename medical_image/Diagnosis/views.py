# views.py - Complete file WITHOUT XAI (heatmaps removed)
# Replaced with: Confidence-Aware Clinical Interpretation (CACI)

import os
import io
import json
import logging
import uuid
import base64
import requests
from datetime import datetime, timedelta

# ✅ PIL Image with alias to avoid conflict with reportlab Image
from PIL import Image as PILImage

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models as tv_models

import numpy as np
from einops import rearrange

from django.shortcuts import render, redirect, get_object_or_404
from django.conf import settings
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate, login as ln, logout
from django.contrib import messages
from django.http import HttpResponse, JsonResponse, FileResponse
from django.views.decorators.csrf import csrf_protect, csrf_exempt
from django.views.decorators.http import require_http_methods
from django.db.models import Count, Q, Avg, Sum
from django.utils import timezone
from django.core.mail import send_mail
from django.template.loader import render_to_string
from django.utils.html import strip_tags
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger

# PDF Generation
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.units import inch, cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.platypus import Image as ReportLabImage
from reportlab.pdfgen import canvas
from reportlab.graphics.shapes import Drawing, Rect, Circle, String
from reportlab.graphics import renderPDF

# Your models
from .models import (
    DiagnosisRecord,
    ContactMessage,
    Notification,
    UserFeedback,
    UserProfile,
    SubscriptionPlan,
    PaymentTransaction,
    ActivityLog,
    SavedAnalysis
)

# ── XAI imports REMOVED ──
# No more gradcam / attention / transformer / panel / clinical heatmaps.

logger = logging.getLogger(__name__)


# ============================================
# DISEASE INFORMATION DATABASE
# ============================================

DISEASE_INFO = {
    'Glioma Tumor': {
        'description': 'Glioma is a type of tumor that originates from glial cells in the brain. It can be benign or malignant and varies in aggressiveness.',
        'severity': 'High',
        'common_symptoms': 'Headaches, seizures, cognitive changes, motor deficits, vision problems',
        'standard_treatment': 'Surgery, radiation therapy, chemotherapy, targeted therapy'
    },
    'Meningioma': {
        'description': 'Meningioma is a typically benign tumor that arises from the meninges, the protective membranes surrounding the brain and spinal cord.',
        'severity': 'Medium',
        'common_symptoms': 'Headaches, vision changes, seizures, weakness, personality changes',
        'standard_treatment': 'Surgery, radiation therapy, observation for small lesions'
    },
    'Pituitary Tumor': {
        'description': 'Pituitary tumors are abnormal growths in the pituitary gland that can affect hormone production and cause various endocrine disturbances.',
        'severity': 'Medium',
        'common_symptoms': 'Hormonal imbalances, headaches, vision problems, fatigue, weight changes',
        'standard_treatment': 'Surgery, medication (dopamine agonists), radiation therapy'
    },
    'Pneumonia': {
        'description': 'Pneumonia is an infection that inflames the air sacs in the lungs, causing them to fill with fluid or pus, leading to breathing difficulties.',
        'severity': 'High',
        'common_symptoms': 'Cough with phlegm, fever, chills, difficulty breathing, chest pain',
        'standard_treatment': 'Antibiotics, antiviral medication, rest, fluids, oxygen therapy'
    },
    'Tuberculosis': {
        'description': 'Tuberculosis (TB) is a bacterial infection caused by Mycobacterium tuberculosis, primarily affecting the lungs but can spread to other organs.',
        'severity': 'High',
        'common_symptoms': 'Persistent cough, fever, night sweats, weight loss, fatigue, chest pain',
        'standard_treatment': 'Multi-drug antibiotic regimen (4 drugs for 6-9 months)'
    },
    'Fracture': {
        'description': 'A bone fracture is a partial or complete break in the continuity of bone tissue resulting from trauma, stress, or pathological conditions.',
        'severity': 'Medium',
        'common_symptoms': 'Pain, swelling, deformity, inability to bear weight, bruising',
        'standard_treatment': 'Immobilization (casting), surgery (internal/external fixation), physical therapy'
    },
    'Melanoma': {
        'description': 'Melanoma is the most aggressive type of skin cancer, developing from melanocytes that produce skin pigment (melanin).',
        'severity': 'High',
        'common_symptoms': 'Changes in existing moles, new growths, asymmetrical lesions, irregular borders, color variation',
        'standard_treatment': 'Surgical excision, immunotherapy, targeted therapy, radiation'
    },
    'No Tumor': {
        'description': 'No abnormal growth or tumor detected in the analyzed region. Findings are within normal limits.',
        'severity': 'Low',
        'common_symptoms': 'None',
        'standard_treatment': 'Routine monitoring, no intervention required'
    },
    'Normal': {
        'description': 'Normal findings with no signs of abnormality or pathology detected.',
        'severity': 'Low',
        'common_symptoms': 'None',
        'standard_treatment': 'Routine monitoring, no intervention required'
    },
    'No_DR': {
        'description': 'No signs of Diabetic Retinopathy detected. Retinal examination is within normal limits.',
        'severity': 'Low',
        'common_symptoms': 'None',
        'standard_treatment': 'Routine monitoring, regular eye check-ups'
    },
    'Mild': {
        'description': 'Mild Diabetic Retinopathy characterized by microaneurysms and small hemorrhages in the retina.',
        'severity': 'Low',
        'common_symptoms': 'Usually asymptomatic; may have occasional blurred vision',
        'standard_treatment': 'Blood sugar control, regular eye monitoring, lifestyle changes'
    },
    'Moderate': {
        'description': 'Moderate Diabetic Retinopathy with increased microaneurysms, hemorrhages, and early retinal changes.',
        'severity': 'Medium',
        'common_symptoms': 'Blurred vision, floaters, difficulty with night vision',
        'standard_treatment': 'Strict blood sugar control, blood pressure management, possible laser therapy'
    },
    'Severe': {
        'description': 'Severe Diabetic Retinopathy with extensive retinal damage, significant hemorrhages, and risk of vision loss.',
        'severity': 'High',
        'common_symptoms': 'Significant vision loss, large floaters, dark spots, difficulty with daily activities',
        'standard_treatment': 'Immediate ophthalmology referral, intensive therapy, possible surgical intervention'
    },
    'Proliferate_DR': {
        'description': 'Proliferative Diabetic Retinopathy with neovascularization and high risk of severe vision loss.',
        'severity': 'Critical',
        'common_symptoms': 'Severe vision loss, floaters, sudden loss of vision, retinal detachment symptoms',
        'standard_treatment': 'Immediate specialized care, laser photocoagulation, anti-VEGF injections, possible surgery'
    },
    'akiec': {
        'description': 'Actinic Keratosis / Bowen\'s disease - precancerous skin lesions that may develop into squamous cell carcinoma.',
        'severity': 'Medium',
        'common_symptoms': 'Scaly, rough patches on sun-exposed skin, may be red, pink, or brown',
        'standard_treatment': 'Topical medications, cryotherapy, photodynamic therapy, surgical removal'
    },
    'bcc': {
        'description': 'Basal Cell Carcinoma - the most common type of skin cancer, typically slow-growing and rarely metastasizes.',
        'severity': 'Medium',
        'common_symptoms': 'Pearl-like bump, persistent sore, pink or red growth, bleeding or crusting lesion',
        'standard_treatment': 'Surgical excision, Mohs surgery, radiation therapy, topical treatments'
    },
    'bkl': {
        'description': 'Benign Keratosis - common benign skin growths that can resemble precancerous or cancerous lesions.',
        'severity': 'Low',
        'common_symptoms': 'Raised, rough, or wart-like growths on skin, may be brown, black, or flesh-colored',
        'standard_treatment': 'Observation, cryotherapy if bothersome, regular monitoring'
    },
    'df': {
        'description': 'Dermatofibroma - a common benign skin lesion, usually harmless but may require monitoring.',
        'severity': 'Low',
        'common_symptoms': 'Small, firm, brownish or pink growth on skin, may be tender or itchy',
        'standard_treatment': 'Observation, surgical removal if symptomatic'
    },
    'mel': {
        'description': 'Melanoma - the most aggressive skin cancer, developing from melanocytes. Early detection is critical.',
        'severity': 'Critical',
        'common_symptoms': 'Asymmetrical growth with irregular borders, color variation, diameter >6mm, evolving changes',
        'standard_treatment': 'Surgical excision, immunotherapy, targeted therapy, chemotherapy, radiation'
    },
    'nv': {
        'description': 'Nevus (mole) - common benign growth on skin, typically harmless but requires monitoring for changes.',
        'severity': 'Low',
        'common_symptoms': 'Small, round, brown or black growth on skin, usually stable over time',
        'standard_treatment': 'Regular monitoring, dermatologist evaluation for any changes'
    },
    'vasc': {
        'description': 'Vascular lesions - abnormalities of blood vessels in the skin, generally benign but may need evaluation.',
        'severity': 'Low',
        'common_symptoms': 'Red, purple, or pink spots on skin, may be raised or flat',
        'standard_treatment': 'Observation, laser treatment if cosmetically concerning'
    }
}


# ============================================
# DECORATOR FOR LOGIN REQUIRED WITH MESSAGE
# ============================================

def login_required_message(view_func):
    """Custom decorator that shows a message for unauthenticated users"""
    def wrapper(request, *args, **kwargs):
        if not request.user.is_authenticated:
            messages.warning(request, 'Please log in or sign up first to access this page.')
            try:
                return redirect('Diagnosis:login')
            except:
                try:
                    return redirect('login')
                except:
                    return redirect('/login/')
        return view_func(request, *args, **kwargs)
    return wrapper


# ============================================
# NOTIFICATION HELPER FUNCTION
# ============================================

def create_notification(user, notification_type, title, message, link=None):
    """Helper function to create notifications"""
    try:
        existing = Notification.objects.filter(
            user=user,
            notification_type=notification_type,
            title=title,
            created_at__gte=timezone.now() - timedelta(minutes=5)
        ).exists()

        if existing:
            return False

        notification = Notification.objects.create(
            user=user,
            notification_type=notification_type,
            title=title,
            message=message,
            link=link
        )

        try:
            from .models import ActivityLog
            ActivityLog.objects.create(
                user=user,
                action_type='other',
                description=f'Notification created: {title}',
                related_record_id=notification.id,
                related_model='Notification'
            )
        except:
            pass

        return True
    except Exception as e:
        print(f"Error creating notification: {e}")
        return False


# ============================================
# GET USER PROFILE HELPER
# ============================================

def get_user_profile(user):
    """Get or create user profile"""
    try:
        return user.profile
    except UserProfile.DoesNotExist:
        return UserProfile.objects.create(user=user)


# ============================================
# LOGIN VIEW
# ============================================

@csrf_protect
def user_login(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')

        if not username or not password:
            messages.error(request, 'Please enter both username and password.')
            return render(request, 'Diagnosis/login.html')

        user = authenticate(request, username=username, password=password)

        if user is not None:
            ln(request, user)
            messages.success(request, f'Welcome back, {user.username}!')

            profile = get_user_profile(user)

            create_notification(
                user,
                'other',
                'Welcome Back',
                f'Welcome back, {user.get_full_name() or user.username}! You have successfully logged in.',
                '/dashboard/'
            )

            next_url = request.GET.get('next')
            if next_url:
                return redirect(next_url)

            try:
                return redirect('Diagnosis:dashboard')
            except:
                try:
                    return redirect('dashboard')
                except:
                    return redirect('/dashboard/')
        else:
            messages.error(request, 'Invalid username or password. Please try again.')
            return render(request, 'Diagnosis/login.html', {'username': username})

    return render(request, 'Diagnosis/login.html')


# ============================================
# REGISTER VIEW
# ============================================

@csrf_protect
def register(request):
    if request.method == 'POST':
        first_name = request.POST.get('first_name', '').strip()
        last_name = request.POST.get('last_name', '').strip()
        username = request.POST.get('username', '').strip()
        email = request.POST.get('email', '').strip()
        password = request.POST.get('password', '').strip()
        role = request.POST.get('role', 'student')
        institution = request.POST.get('institution', '').strip()
        phone = request.POST.get('phone', '').strip()

        errors = []

        if not first_name:
            errors.append('First name is required.')
        if not last_name:
            errors.append('Last name is required.')
        if not username:
            errors.append('Username is required.')
        if not email:
            errors.append('Email is required.')
        if len(password) < 8:
            errors.append('Password must be at least 8 characters.')

        if username and User.objects.filter(username=username).exists():
            errors.append('Username already taken. Please choose another.')

        if email and User.objects.filter(email=email).exists():
            errors.append('Email already registered. Please use another.')

        if errors:
            for error in errors:
                messages.error(request, error)
            return render(request, 'Diagnosis/register.html', {
                'first_name': first_name,
                'last_name': last_name,
                'username': username,
                'email': email,
                'role': role,
                'institution': institution,
                'phone': phone,
            })

        try:
            user = User.objects.create_user(
                username=username,
                email=email,
                password=password,
                first_name=first_name,
                last_name=last_name
            )

            profile = UserProfile.objects.create(
                user=user,
                role=role,
                institution=institution,
                phone=phone
            )

            messages.success(request, f'Account created successfully! Welcome, {first_name}!')

            ln(request, user)
            messages.info(request, 'You have been automatically logged in.')

            create_notification(
                user,
                'welcome',
                'Welcome to X-HViT',
                f'Welcome {first_name}! Thank you for joining X-HViT. Start your first diagnosis today.',
                '/diagnose/'
            )

            try:
                return redirect('Diagnosis:dashboard')
            except:
                try:
                    return redirect('dashboard')
                except:
                    return redirect('/dashboard/')

        except Exception as e:
            messages.error(request, f'Error creating account: {str(e)}')
            return render(request, 'Diagnosis/register.html', {
                'first_name': first_name,
                'last_name': last_name,
                'username': username,
                'email': email,
                'role': role,
                'institution': institution,
                'phone': phone,
            })

    return render(request, 'Diagnosis/register.html')


# ============================================
# LOGOUT VIEW
# ============================================

def user_logout(request):
    logout(request)
    messages.info(request, 'You have been logged out successfully.')
    try:
        return redirect('Diagnosis:index')
    except:
        try:
            return redirect('index')
        except:
            return redirect('/')


# ============================================
# MAIN PAGE VIEWS (Public - No login required)
# ============================================

def index(request):
    return render(request, 'Diagnosis/index.html')

def about(request):
    return render(request, 'Diagnosis/about.html')

def terms(request):
    return render(request, 'Diagnosis/terms.html')

def services(request):
    return render(request, 'Diagnosis/services.html')

def contact(request):
    """Contact page with form submission handling"""
    if request.method == 'POST':
        first_name = request.POST.get('first_name', '').strip()
        last_name = request.POST.get('last_name', '').strip()
        email = request.POST.get('email', '').strip()
        category = request.POST.get('category', 'general')
        subject = request.POST.get('subject', '').strip()
        message = request.POST.get('message', '').strip()

        errors = []
        if not first_name:
            errors.append('First name is required.')
        if not last_name:
            errors.append('Last name is required.')
        if not email:
            errors.append('Email is required.')
        if not subject:
            errors.append('Subject is required.')
        if not message:
            errors.append('Message is required.')

        if errors:
            messages.error(request, 'Please fix the following errors: ' + ' '.join(errors))
            return render(request, 'Diagnosis/contact.html', {
                'first_name': first_name,
                'last_name': last_name,
                'email': email,
                'category': category,
                'subject': subject,
                'message': message,
            })

        try:
            contact_message = ContactMessage.objects.create(
                user=request.user if request.user.is_authenticated else None,
                first_name=first_name,
                last_name=last_name,
                email=email,
                category=category,
                subject=subject,
                message=message,
                status='new'
            )

            if request.user.is_authenticated:
                create_notification(
                    request.user,
                    'other',
                    'Message Received',
                    f'Your message "{subject}" has been received. We will respond within 2-3 business days.',
                    '/contact/'
                )

            messages.success(request, 'Your message has been sent successfully! We will respond within 2-3 business days.')
            return redirect('Diagnosis:contact')

        except Exception as e:
            messages.error(request, f'An error occurred: {str(e)}')
            return render(request, 'Diagnosis/contact.html', {
                'first_name': first_name,
                'last_name': last_name,
                'email': email,
                'category': category,
                'subject': subject,
                'message': message,
            })

    return render(request, 'Diagnosis/contact.html')

def faq(request):
    return render(request, 'Diagnosis/faq.html')

def documentation(request):
    return render(request, 'Diagnosis/documentation.html')

def modalities(request):
    return render(request, 'Diagnosis/modalities.html')

@login_required
def pricing(request):
    """Pricing page with subscription info"""
    profile = get_user_profile(request.user)
    plans = SubscriptionPlan.objects.filter(is_active=True)

    unread_count = 0
    if request.user.is_authenticated:
        try:
            from .models import Notification
            unread_count = Notification.objects.filter(
                user=request.user,
                is_read=False
            ).count()
        except:
            unread_count = 0

    context = {
        'profile': profile,
        'current_plan': profile.subscription_plan,
        'is_subscribed': profile.is_subscribed,
        'subscription_expiry': profile.subscription_expiry,
        'plans': plans,
        'unread_notifications_count': unread_count,
    }
    return render(request, 'Diagnosis/pricing.html', context)

def privacy(request):
    return render(request, 'Diagnosis/privacy.html')

def support(request):
    return render(request, 'Diagnosis/support.html')

def team(request):
    return render(request, 'Diagnosis/team.html')


# ============================================
# REPORTS VIEW
# ============================================

@login_required
def reports_view(request):
    """Display all diagnostic reports with filtering"""
    try:
        profile = get_user_profile(request.user)
        records = DiagnosisRecord.objects.filter(user=request.user).order_by('-created_at')

        total_reports = records.filter(report_generated=True).count()
        total_diagnoses = records.count()

        now = timezone.now()
        start_of_month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        monthly_reports = records.filter(created_at__gte=start_of_month, report_generated=True).count()

        total_downloads = records.aggregate(total=Sum('download_count'))['total'] or 0

        report_data = []
        for record in records:
            modality_name = record.get_modality_display() if record.modality else 'Unknown'
            badge_class = get_modality_badge_class(record.modality)
            icon = get_modality_icon(record.modality)
            bg_color = get_confidence_color(record.confidence)
            status = record.status or 'Completed'

            report_data.append({
                'id': record.id,
                'case_id': record.case_id,
                'modality': modality_name,
                'badge_class': badge_class,
                'diagnosis': record.diagnosis or 'Unknown',
                'confidence': record.confidence or 0,
                'created_at': record.created_at,
                'icon': icon,
                'bg_color': bg_color,
                'status': status,
            })

        context = {
            'user': request.user,
            'profile': profile,
            'role': profile.get_role_display(),
            'subscription': profile.subscription_plan,
            'is_subscribed': profile.is_subscribed,
            'reports': report_data,
            'total_reports': total_reports,
            'total_diagnoses': total_diagnoses,
            'monthly_reports': monthly_reports,
            'total_downloads': total_downloads,
        }

        return render(request, 'Diagnosis/reports.html', context)

    except Exception as e:
        logger.error(f"Error in reports_view: {str(e)}")
        messages.error(request, 'Error loading reports. Please try again.')
        return render(request, 'Diagnosis/reports.html', {
            'reports': [],
            'total_reports': 0,
            'total_diagnoses': 0,
            'monthly_reports': 0,
            'total_downloads': 0,
        })


# ============================================
# DASHBOARD VIEW
# ============================================

@login_required_message
def dashboard_view(request):
    """Dashboard page - Main landing after login with dynamic data"""
    if not request.user.is_authenticated:
        messages.warning(request, 'Please log in to access the dashboard.')
        return redirect('Diagnosis:login')

    profile = get_user_profile(request.user)
    records = DiagnosisRecord.objects.filter(user=request.user)

    total_analyses = records.count()

    week_ago = timezone.now() - timedelta(days=7)
    weekly_analyses = records.filter(created_at__gte=week_ago).count()

    avg_confidence = records.aggregate(avg=Avg('confidence'))['avg'] or 0

    total_reports = records.filter(report_generated=True).count()

    new_reports = records.filter(report_generated=True, created_at__gte=week_ago).count()

    avg_inference_time = 0
    records_with_time = records.exclude(inference_time='')
    if records_with_time.exists():
        times = []
        for r in records_with_time:
            try:
                if r.inference_time:
                    times.append(float(r.inference_time) / 1000)
            except (ValueError, TypeError):
                pass
        if times:
            avg_inference_time = sum(times) / len(times)

    recent_diagnoses = records.order_by('-created_at')[:5]

    recent_activity = []
    for record in records.order_by('-created_at')[:5]:
        confidence_display = f"{record.confidence:.1f}" if record.confidence else "0.0"
        recent_activity.append({
            'type': 'diagnosis',
            'text': f'New diagnosis completed: <strong>{record.diagnosis}</strong> ({confidence_display}%)',
            'time': record.created_at
        })
        if record.report_generated:
            recent_activity.append({
                'type': 'report',
                'text': f'Report generated for <strong>{record.case_id}</strong>',
                'time': record.created_at
            })
    recent_activity = recent_activity[:5]

    modality_stats = records.values('modality').annotate(count=Count('id'))
    total = sum(item['count'] for item in modality_stats)
    modality_distribution = []
    modality_icons = {
        'MRI': '🧠', 'CT': '🫁', 'X-ray': '🩻', 'Micro': '🔭',
        'auto': '🤖', 'Unknown': '📊',
    }
    modality_colors = {
        'MRI': 'var(--blue)', 'CT': 'var(--teal)', 'X-ray': 'var(--violet)',
        'Micro': 'var(--yellow)', 'auto': 'var(--teal)', 'Unknown': 'var(--text-muted)',
    }
    for item in modality_stats:
        name = item['modality'] or 'Unknown'
        percentage = (item['count'] / total * 100) if total > 0 else 0
        modality_distribution.append({
            'name': name,
            'icon': modality_icons.get(name, '📊'),
            'count': item['count'],
            'percentage': percentage,
            'color': modality_colors.get(name, 'var(--blue)')
        })

    modality_accuracy = []
    for modality in records.values('modality').distinct():
        name = modality['modality'] or 'Unknown'
        acc = records.filter(modality=modality['modality']).aggregate(avg=Avg('confidence'))['avg'] or 0
        modality_accuracy.append({
            'name': name,
            'icon': modality_icons.get(name, '📊'),
            'accuracy': acc
        })

    month_ago = timezone.now() - timedelta(days=30)
    recent_avg = records.filter(created_at__gte=month_ago).aggregate(avg=Avg('confidence'))['avg'] or 0
    older_avg = records.filter(created_at__lt=month_ago).aggregate(avg=Avg('confidence'))['avg'] or 0
    confidence_change = recent_avg - older_avg if older_avg > 0 else 0

    is_subscribed = profile.is_subscribed and profile.has_active_subscription()
    subscription_status = 'active' if is_subscribed else 'inactive'

    context = {
        'user': request.user,
        'profile': profile,
        'role': profile.get_role_display(),
        'subscription': profile.subscription_plan,
        'is_subscribed': is_subscribed,
        'subscription_status': subscription_status,
        'subscription_expiry': profile.subscription_expiry,
        'total_analyses': total_analyses,
        'weekly_analyses': weekly_analyses,
        'avg_confidence': avg_confidence,
        'confidence_change': confidence_change,
        'total_reports': total_reports,
        'new_reports': new_reports,
        'avg_inference_time': avg_inference_time,
        'recent_diagnoses': recent_diagnoses,
        'recent_activity': recent_activity,
        'modality_distribution': modality_distribution,
        'modality_accuracy': modality_accuracy,
    }

    return render(request, 'Diagnosis/dashboard.html', context)


# ============================================
# PROFILE VIEW
# ============================================

@login_required
def profile_view(request):
    """User profile management page"""
    profile = get_user_profile(request.user)

    if request.method == 'POST':
        first_name = request.POST.get('first_name', '').strip()
        last_name = request.POST.get('last_name', '').strip()
        phone = request.POST.get('phone', '').strip()
        institution = request.POST.get('institution', '').strip()
        role = request.POST.get('role', profile.role)

        user = request.user
        user.first_name = first_name
        user.last_name = last_name
        user.save()

        profile.phone = phone
        profile.institution = institution
        profile.role = role

        if request.FILES.get('profile_picture'):
            profile.profile_picture = request.FILES.get('profile_picture')

        profile.save()
        messages.success(request, 'Profile updated successfully!')
        return redirect('Diagnosis:profile')

    context = {
        'user': request.user,
        'profile': profile,
        'role_display': profile.get_role_display(),
    }
    return render(request, 'Diagnosis/profile.html', context)


# ============================================
# REPORT DETAIL VIEW (XAI removed)
# ============================================

@login_required
def report_detail_view(request, report_id):
    """View a single report detail"""
    try:
        profile = get_user_profile(request.user)
        record = get_object_or_404(DiagnosisRecord, id=report_id, user=request.user)

        disease_info = DISEASE_INFO.get(record.diagnosis, {})

        probabilities = []
        if record.probabilities:
            try:
                if isinstance(record.probabilities, list):
                    probabilities = record.probabilities
                elif isinstance(record.probabilities, str):
                    probabilities = json.loads(record.probabilities)
            except:
                probabilities = []

        if not probabilities:
            probabilities = [
                [record.diagnosis, record.confidence, '#00e5b0']
            ]

        similar_cases = DiagnosisRecord.objects.filter(
            user=request.user,
            diagnosis=record.diagnosis
        ).exclude(id=record.id).order_by('-confidence')[:5]

        # ── Clinical summary (replaces clinical heatmap) ──
        clinical_summary = getattr(record, 'clinical_summary', None)

        # Fallback: build on the fly from probabilities + confidence
        if not clinical_summary and probabilities:
            sorted_probs = sorted(probabilities, key=lambda x: x[1], reverse=True)[:3]
            conf = record.confidence or 0
            if conf >= 90:
                strength = "Strong positive finding"
            elif conf >= 75:
                strength = "Moderate finding"
            elif conf >= 60:
                strength = "Weak finding — recommend human review"
            else:
                strength = "Inconclusive — recommend human review"

            margin = (sorted_probs[0][1] - sorted_probs[1][1]) if len(sorted_probs) > 1 else 100.0
            if margin >= 50:
                margin_label = "clear margin"
            elif margin >= 20:
                margin_label = "moderate margin"
            else:
                margin_label = "narrow margin — consider differential"

            clinical_summary = {
                'primary': record.diagnosis,
                'confidence': conf,
                'strength': strength,
                'differential': sorted_probs,
                'margin': round(margin, 1),
                'margin_label': margin_label,
                'is_positive': record.is_positive,
            }

        context = {
            'user': request.user,
            'profile': profile,
            'role': profile.get_role_display(),
            'subscription': profile.subscription_plan,
            'is_subscribed': profile.is_subscribed,
            'record': record,
            'disease_info': disease_info,
            'probabilities': probabilities,
            'similar_cases': similar_cases,
            'clinical_summary': clinical_summary,   # ← replaces clinical_media_url
        }
        return render(request, 'Diagnosis/report_detail.html', context)

    except DiagnosisRecord.DoesNotExist:
        messages.error(request, 'Report not found.')
        return redirect('Diagnosis:reports')
    except Exception as e:
        logger.error(f"Error in report_detail_view: {str(e)}")
        messages.error(request, 'Error loading report details.')
        return redirect('Diagnosis:reports')


# ============================================
# DOWNLOAD REPORT VIEW
# ============================================

@login_required
@require_http_methods(['GET'])
def download_report_view(request, report_id):
    """Download report as PDF"""
    try:
        profile = get_user_profile(request.user)
        record = get_object_or_404(DiagnosisRecord, id=report_id, user=request.user)

        try:
            pdf_content = generate_report_pdf(record)
        except Exception as pdf_error:
            logger.error(f"PDF generation error: {str(pdf_error)}")
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return JsonResponse({'error': f'PDF generation failed: {str(pdf_error)}'}, status=500)
            messages.error(request, f'Error generating PDF: {str(pdf_error)}')
            return redirect('Diagnosis:reports')

        try:
            if hasattr(record, 'download_count'):
                record.download_count = record.download_count + 1
                record.save(update_fields=['download_count'])
                profile.total_reports += 1
                profile.save()
        except Exception as e:
            logger.warning(f"Could not update download_count: {e}")

        if not record.report_generated:
            record.report_generated = True
            record.save(update_fields=['report_generated'])

        response = HttpResponse(pdf_content, content_type='application/pdf')
        response['Content-Disposition'] = f'attachment; filename="report_{record.case_id}.pdf"'
        response['Content-Length'] = len(pdf_content)

        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return response

        messages.success(request, f'PDF for case {record.case_id} downloaded successfully!')
        return redirect('Diagnosis:reports')

    except DiagnosisRecord.DoesNotExist:
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return JsonResponse({'error': 'Report not found'}, status=404)
        messages.error(request, 'Report not found.')
        return redirect('Diagnosis:reports')

    except Exception as e:
        logger.error(f"Error in download_report_view: {str(e)}")
        import traceback
        traceback.print_exc()
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return JsonResponse({'error': str(e)}, status=500)
        messages.error(request, f'Error generating PDF: {str(e)}')
        return redirect('Diagnosis:reports')


# ============================================
# GENERATE REPORT PDF (XAI removed)
# ============================================

def generate_report_pdf(record):
    """Generate PDF content for a report using DiagnosisRecord"""
    try:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72,
        )

        styles = getSampleStyleSheet()

        title_style = ParagraphStyle(
            'CustomTitle', parent=styles['Heading1'], fontSize=24,
            textColor=colors.HexColor('#00e5b0'),
            alignment=TA_CENTER, spaceAfter=12,
        )
        heading_style = ParagraphStyle(
            'CustomHeading', parent=styles['Heading2'], fontSize=16,
            textColor=colors.HexColor('#f0f4fc'), spaceAfter=8,
        )
        body_style = ParagraphStyle(
            'CustomBody', parent=styles['Normal'], fontSize=12,
            textColor=colors.HexColor('#94a9c9'), spaceAfter=6,
        )

        story = []
        story.append(Paragraph(f"Diagnostic Report - {record.case_id}", title_style))
        story.append(Spacer(1, 12))
        story.append(Paragraph(f"Generated: {record.created_at.strftime('%Y-%m-%d %H:%M')}", body_style))
        story.append(Spacer(1, 12))

        story.append(Paragraph("Report Details", heading_style))
        story.append(Spacer(1, 6))

        details_data = [
            ['Case ID', str(record.case_id)],
            ['Modality', str(record.get_modality_display() or 'Unknown')],
            ['Diagnosis', str(record.diagnosis or 'N/A')],
            ['Confidence', f"{record.confidence or 0:.1f}%"],
            ['Status', str(record.status or 'Completed')],
            ['Inference Time', str(record.inference_time or 'N/A')],
        ]

        if record.probabilities:
            try:
                probs = record.probabilities
                if isinstance(probs, list) and len(probs) > 0:
                    prob_text = ', '.join([f"{p[0]}: {p[1]:.1f}%" for p in probs[:3]])
                    details_data.append(['Top Probabilities', str(prob_text)])
            except:
                pass

        details_table = Table(details_data, colWidths=[150, 300])
        details_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#0b121f')),
            ('BACKGROUND', (1, 0), (1, -1), colors.HexColor('#0f1a2a')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#f0f4fc')),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('LEFTPADDING', (0, 0), (-1, -1), 12),
            ('RIGHTPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#131e30')),
        ]))
        story.append(details_table)
        story.append(Spacer(1, 20))

        # ── Clinical Summary (replaces heatmap) ──
        clinical_summary = getattr(record, 'clinical_summary', None)

        # Fallback: build from probabilities if not stored
        if not clinical_summary and record.probabilities:
            try:
                probs = record.probabilities
                if isinstance(probs, str):
                    probs = json.loads(probs)
                sorted_probs = sorted(probs, key=lambda x: x[1], reverse=True)[:3]
                conf = record.confidence or 0
                if conf >= 90:
                    strength = "Strong positive finding"
                elif conf >= 75:
                    strength = "Moderate finding"
                elif conf >= 60:
                    strength = "Weak finding — recommend human review"
                else:
                    strength = "Inconclusive — recommend human review"
                margin = (sorted_probs[0][1] - sorted_probs[1][1]) if len(sorted_probs) > 1 else 100.0
                margin_label = "clear margin" if margin >= 50 else ("moderate margin" if margin >= 20 else "narrow margin — consider differential")
                clinical_summary = {
                    'primary': record.diagnosis,
                    'confidence': conf,
                    'strength': strength,
                    'differential': sorted_probs,
                    'margin': round(margin, 1),
                    'margin_label': margin_label,
                    'is_positive': record.is_positive,
                }
            except Exception:
                clinical_summary = None

        if clinical_summary:
            story.append(Paragraph("Clinical Summary", heading_style))
            story.append(Spacer(1, 6))

            summary_data = [
                ['Primary Finding', str(clinical_summary.get('primary', 'N/A'))],
                ['Confidence', f"{clinical_summary.get('confidence', 0)}%"],
                ['Finding Strength', str(clinical_summary.get('strength', 'N/A'))],
                ['Decision Margin', str(clinical_summary.get('margin_label', 'N/A'))],
            ]
            summary_table = Table(summary_data, colWidths=[150, 300])
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#0b121f')),
                ('BACKGROUND', (1, 0), (1, -1), colors.HexColor('#0f1a2a')),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#f0f4fc')),
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 11),
                ('TOPPADDING', (0, 0), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                ('LEFTPADDING', (0, 0), (-1, -1), 12),
                ('RIGHTPADDING', (0, 0), (-1, -1), 12),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#131e30')),
            ]))
            story.append(summary_table)
            story.append(Spacer(1, 12))

            diff = clinical_summary.get('differential', [])
            if diff:
                diff_lines = "<br/>".join([f"• {d[0]}: {d[1]}%" for d in diff])
                story.append(Paragraph("Differential Diagnosis", heading_style))
                story.append(Spacer(1, 4))
                story.append(Paragraph(diff_lines, body_style))
                story.append(Spacer(1, 20))

        disease_info = DISEASE_INFO.get(record.diagnosis)
        if disease_info:
            story.append(Paragraph("Disease Information", heading_style))
            story.append(Spacer(1, 6))
            info_data = [
                ['Description', str(disease_info.get('description', 'N/A'))],
                ['Severity', str(disease_info.get('severity', 'N/A'))],
                ['Common Symptoms', str(disease_info.get('common_symptoms', 'N/A'))],
                ['Standard Treatment', str(disease_info.get('standard_treatment', 'N/A'))],
            ]
            info_table = Table(info_data, colWidths=[150, 300])
            info_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#0b121f')),
                ('BACKGROUND', (1, 0), (1, -1), colors.HexColor('#0f1a2a')),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#f0f4fc')),
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 11),
                ('TOPPADDING', (0, 0), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                ('LEFTPADDING', (0, 0), (-1, -1), 12),
                ('RIGHTPADDING', (0, 0), (-1, -1), 12),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#131e30')),
            ]))
            story.append(info_table)
            story.append(Spacer(1, 20))

        if record.description:
            story.append(Paragraph("Clinical Notes", heading_style))
            story.append(Spacer(1, 6))
            story.append(Paragraph(str(record.description), body_style))
            story.append(Spacer(1, 20))

        story.append(Spacer(1, 30))
        story.append(Paragraph(
            "This report was generated automatically by X-HViT Medical AI System.",
            ParagraphStyle(
                'Footer', parent=styles['Normal'], fontSize=10,
                textColor=colors.HexColor('#4e6688'), alignment=TA_CENTER,
            )
        ))

        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()

    except Exception as e:
        logger.error(f"Error generating PDF: {str(e)}")
        import traceback
        traceback.print_exc()
        raise Exception(f"PDF generation failed: {str(e)}")


# ============================================
# HELPER FUNCTIONS
# ============================================

def get_modality_badge_class(modality):
    modality_map = {
        'MRI': 'badge-blue', 'CT': 'badge-teal', 'X-ray': 'badge-violet',
        'Micro': 'badge-yellow', 'auto': 'badge-red',
    }
    return modality_map.get(modality, 'badge-blue')


def get_modality_icon(modality):
    icon_map = {
        'MRI': '🧠', 'CT': '🫁', 'X-ray': '🩻', 'Micro': '🔭', 'auto': '🤖',
    }
    return icon_map.get(modality, '📊')


def get_confidence_color(confidence):
    if confidence >= 90:
        return 'rgba(0,229,176,0.06)'
    elif confidence >= 75:
        return 'rgba(58,155,255,0.06)'
    elif confidence >= 50:
        return 'rgba(255,201,71,0.06)'
    else:
        return 'rgba(255,85,117,0.06)'


# ============================================
# OTHER PAGE VIEWS
# ============================================

@login_required_message
def result(request):
    profile = get_user_profile(request.user)
    context = {
        'user': request.user, 'profile': profile,
        'role': profile.get_role_display(),
        'subscription': profile.subscription_plan,
        'is_subscribed': profile.is_subscribed,
    }
    return render(request, 'Diagnosis/result.html', context)

@login_required_message
def report(request):
    profile = get_user_profile(request.user)
    context = {
        'user': request.user, 'profile': profile,
        'role': profile.get_role_display(),
        'subscription': profile.subscription_plan,
        'is_subscribed': profile.is_subscribed,
    }
    return render(request, 'Diagnosis/report.html', context)


# ============================================
# HISTORY PAGE VIEW
# ============================================

@login_required_message
def history_page(request):
    """History page with all diagnosis records and visualization data"""
    profile = get_user_profile(request.user)
    records = DiagnosisRecord.objects.filter(user=request.user).order_by('-created_at')

    total_records = records.count()
    completed_count = records.filter(status='Completed').count()
    review_count = records.filter(status='Review').count()
    positive_count = records.filter(is_positive=True).count()

    # Map 'auto' modality to actual modality
    for record in records:
        if record.modality == 'auto' or record.modality is None or record.modality == '':
            disease_key = getattr(record, 'disease_category', '')
            modality_map = {
                'brain_tumor': 'MRI', 'bone_fracture': 'X-ray',
                'tuberculosis': 'X-ray', 'diabetic_retinopathy': 'Microscopic',
                'skin_cancer': 'Microscopic', 'pneumonia': 'X-ray',
                'fracatlas': 'X-ray',
            }
            mapped_mod = modality_map.get(disease_key, 'X-ray')
            record.modality = mapped_mod
            record.save(update_fields=['modality'])

    # ── Disease distribution ──
    disease_mapping = {
        'Glioma Tumor': 'Brain Tumor', 'Meningioma': 'Brain Tumor',
        'Pituitary Tumor': 'Brain Tumor', 'No Tumor': 'No Tumor',
        'Normal': 'Normal', 'Fracture': 'Fracture',
        'Bone Fracture': 'Fracture', 'No Fracture': 'Normal',
        'Tuberculosis': 'Tuberculosis', 'Pneumonia': 'Pneumonia',
        'No_DR': 'Normal', 'Mild': 'Diabetic Retinopathy',
        'Moderate': 'Diabetic Retinopathy', 'Severe': 'Diabetic Retinopathy',
        'Proliferate_DR': 'Diabetic Retinopathy',
        'akiec': 'Skin Cancer', 'bcc': 'Skin Cancer', 'bkl': 'Skin Cancer',
        'df': 'Skin Cancer', 'mel': 'Skin Cancer', 'nv': 'Skin Cancer',
        'vasc': 'Skin Cancer', 'Melanoma': 'Skin Cancer',
    }

    disease_groups = {}
    for record in records:
        diagnosis = record.diagnosis
        disease_name = disease_mapping.get(diagnosis, diagnosis)
        if disease_name not in disease_groups:
            disease_groups[disease_name] = 0
        disease_groups[disease_name] += 1

    sorted_diseases = sorted(disease_groups.items(), key=lambda x: x[1], reverse=True)
    disease_labels = [d[0] for d in sorted_diseases[:7]]
    disease_values = [d[1] for d in sorted_diseases[:7]]

    if not disease_labels:
        disease_labels = ['No Data']; disease_values = [1]

    # ── Monthly trends ──
    end_date = timezone.now()
    start_date = end_date - timedelta(days=180)
    monthly_data = records.filter(
        created_at__gte=start_date
    ).extra({'month': "strftime('%%Y-%%m', created_at)"}).values('month').annotate(count=Count('id')).order_by('month')

    months, counts = [], []
    current = start_date.replace(day=1)
    while current <= end_date:
        month_key = current.strftime('%Y-%m')
        months.append(current.strftime('%b %Y'))
        match = next((item for item in monthly_data if item['month'] == month_key), None)
        counts.append(match['count'] if match else 0)
        current = (current.replace(day=28) + timedelta(days=4)).replace(day=1)

    if not months:
        months = ['No Data']; counts = [0]

    # ── Confidence distribution ──
    high_conf = records.filter(confidence__gte=80).count()
    medium_conf = records.filter(confidence__gte=60, confidence__lt=80).count()
    low_conf = records.filter(confidence__lt=60).count()
    confidence_labels = ['High (80-100%)', 'Medium (60-79%)', 'Low (0-59%)']
    confidence_values = [high_conf, medium_conf, low_conf]

    # ── Modality distribution ──
    modality_counts = {'MRI': 0, 'CT': 0, 'X-ray': 0, 'Microscopic': 0}
    for record in records:
        mod = record.modality
        if mod in modality_counts:
            modality_counts[mod] += 1
        elif mod == 'auto' or mod is None or mod == '':
            disease_key = getattr(record, 'disease_category', '')
            auto_map = {
                'brain_tumor': 'MRI', 'bone_fracture': 'X-ray',
                'tuberculosis': 'X-ray', 'diabetic_retinopathy': 'Microscopic',
                'skin_cancer': 'Microscopic', 'pneumonia': 'X-ray',
                'fracatlas': 'X-ray',
            }
            mapped_mod = auto_map.get(disease_key, 'X-ray')
            modality_counts[mapped_mod] += 1

    modality_labels = [k for k, v in modality_counts.items() if v > 0]
    modality_values = [v for k, v in modality_counts.items() if v > 0]
    if not modality_labels:
        modality_labels = ['No Data']; modality_values = [1]

    # ── Status distribution ──
    status_data = records.values('status').annotate(count=Count('status'))
    status_labels = [item['status'] for item in status_data if item['status']]
    status_values = [item['count'] for item in status_data if item['status']]
    if not status_labels:
        status_labels = ['No Data']; status_values = [1]

    # ── Timeline ──
    timeline_labels, timeline_values, timeline_diseases, timeline_dates = [], [], [], []
    for i, record in enumerate(records[:30], 1):
        timeline_labels.append(f"#{i}")
        timeline_values.append(record.confidence if record.confidence else 50)
        disease_name = disease_mapping.get(record.diagnosis, record.diagnosis)
        timeline_diseases.append(disease_name)
        timeline_dates.append(record.created_at.strftime('%b %d, %Y'))

    # ── Pagination ──
    paginator = Paginator(records, 10)
    page = request.GET.get('page')
    try:
        paginated_records = paginator.page(page)
    except PageNotAnInteger:
        paginated_records = paginator.page(1)
    except EmptyPage:
        paginated_records = paginator.page(paginator.num_pages)

    record_data = []
    for record in paginated_records:
        probabilities = []
        if record.probabilities:
            try:
                if isinstance(record.probabilities, list):
                    probabilities = record.probabilities
                elif isinstance(record.probabilities, str):
                    probabilities = json.loads(record.probabilities)
            except:
                probabilities = []

        record_data.append({
            'id': record.id,
            'case_id': record.case_id,
            'modality': record.modality or 'Unknown',
            'diagnosis': record.diagnosis or 'Unknown',
            'confidence': record.confidence or 0,
            'status': record.status or 'Completed',
            'inference_time': record.inference_time or 'N/A',
            'created_at': record.created_at,
            'probabilities': probabilities,
            'is_positive': record.is_positive,
            'report_generated': record.report_generated,
        })

    context = {
        'user': request.user, 'profile': profile,
        'role': profile.get_role_display(),
        'subscription': profile.subscription_plan,
        'is_subscribed': profile.is_subscribed,
        'records': paginated_records, 'record_data': record_data,
        'total_records': total_records,
        'completed_count': completed_count,
        'review_count': review_count,
        'positive_count': positive_count,
        'avg_confidence': records.aggregate(avg=Avg('confidence'))['avg'] or 0,
        'modality_count': len([m for m in modality_counts.values() if m > 0]),
        'disease_distribution': json.dumps({'labels': disease_labels, 'values': disease_values}),
        'monthly_trends': json.dumps({'dates': months, 'counts': counts}),
        'confidence_distribution': json.dumps({'labels': confidence_labels, 'values': confidence_values}),
        'modality_distribution': json.dumps({'labels': modality_labels, 'values': modality_values}),
        'status_distribution': json.dumps({'labels': status_labels, 'values': status_values}),
        'timeline_data': json.dumps({
            'labels': timeline_labels, 'values': timeline_values,
            'diseases': timeline_diseases, 'dates': timeline_dates
        }),
    }

    return render(request, 'Diagnosis/history.html', context)


# ============================================
# DIAGNOSE PAGE VIEW
# ============================================

@login_required_message
def diagnose_page(request):
    profile = get_user_profile(request.user)

    if profile.subscription_plan == 'free':
        diagnoses_used = DiagnosisRecord.objects.filter(user=request.user).count()
        if diagnoses_used >= 10:
            messages.warning(request, 'You have reached the free plan limit (10 diagnoses). Please upgrade to continue.')
            return redirect('Diagnosis:pricing')

    context = {
        'user': request.user, 'profile': profile,
        'role': profile.get_role_display(),
        'subscription': profile.subscription_plan,
        'is_subscribed': profile.is_subscribed,
    }
    return render(request, 'Diagnosis/diagnose.html', context)


@login_required
def home(request):
    try:
        return redirect('Diagnosis:dashboard')
    except:
        try:
            return redirect('dashboard')
        except:
            return redirect('/dashboard/')


# ============================================
# API VIEWS
# ============================================

@login_required
def get_diagnosis_history(request):
    profile = get_user_profile(request.user)
    records = DiagnosisRecord.objects.filter(user=request.user).order_by('-created_at')
    data = [{
        'id': r.id, 'case_id': r.case_id, 'modality': r.modality,
        'diagnosis': r.diagnosis, 'confidence': r.confidence,
        'status': r.status, 'report_generated': r.report_generated,
        'is_positive': r.is_positive,
        'created_at': r.created_at.strftime('%Y-%m-%d %H:%M:%S'),
        'description': r.description or ''
    } for r in records]
    return JsonResponse({
        'status': 'success', 'records': data,
        'profile': {
            'role': profile.get_role_display(),
            'subscription': profile.subscription_plan,
            'is_subscribed': profile.is_subscribed,
            'total_diagnoses': profile.total_diagnoses,
            'total_reports': profile.total_reports,
        }
    })


@login_required
@require_http_methods(["DELETE"])
def delete_diagnosis(request, record_id):
    try:
        record = DiagnosisRecord.objects.get(id=record_id, user=request.user)
        record.delete()
        return JsonResponse({'status': 'success', 'message': 'Record deleted successfully'})
    except DiagnosisRecord.DoesNotExist:
        return JsonResponse({'status': 'error', 'message': 'Record not found'}, status=404)
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)


@login_required
@csrf_exempt
@require_http_methods(["POST"])
def clear_history(request):
    try:
        count = DiagnosisRecord.objects.filter(user=request.user).delete()
        return JsonResponse({'status': 'success', 'message': f'Deleted {count[0]} records successfully'})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)


@login_required
def get_record_detail(request, record_id):
    try:
        record = DiagnosisRecord.objects.get(id=record_id, user=request.user)
        data = {
            'id': record.id, 'case_id': record.case_id,
            'modality': record.modality, 'diagnosis': record.diagnosis,
            'confidence': record.confidence, 'status': record.status,
            'report_generated': record.report_generated,
            'is_positive': record.is_positive,
            'created_at': record.created_at.strftime('%Y-%m-%d %H:%M:%S'),
            'description': record.description or '',
            'image_path': record.image_path or '',
            'report_path': record.report_path or '',
            'probabilities': record.probabilities or [],
            'clinical_summary': getattr(record, 'clinical_summary', None),
        }
        return JsonResponse({'status': 'success', 'record': data})
    except DiagnosisRecord.DoesNotExist:
        return JsonResponse({'status': 'error', 'message': 'Record not found'}, status=404)


# ============================================
# NOTIFICATIONS VIEWS
# ============================================

@login_required
def notifications_view(request):
    profile = get_user_profile(request.user)
    notifications = Notification.objects.filter(user=request.user).order_by('-created_at')

    if request.GET.get('mark_all_read'):
        notifications.filter(is_read=False).update(is_read=True)
        messages.success(request, 'All notifications marked as read.')
        return redirect('Diagnosis:notifications')

    unread_count = notifications.filter(is_read=False).count()
    total_count = notifications.count()
    read_count = total_count - unread_count

    context = {
        'user': request.user, 'profile': profile,
        'role': profile.get_role_display(),
        'subscription': profile.subscription_plan,
        'is_subscribed': profile.is_subscribed,
        'notifications': notifications,
        'unread_count': unread_count,
        'read_count': read_count,
        'total_count': total_count,
    }
    return render(request, 'Diagnosis/notifications.html', context)


@login_required
def mark_notification_read(request, notification_id):
    notification = get_object_or_404(Notification, id=notification_id, user=request.user)
    notification.is_read = True
    notification.save()
    if request.headers.get('x-requested-with') == 'XMLHttpRequest':
        return JsonResponse({'status': 'success'})
    messages.success(request, 'Notification marked as read.')
    return redirect('Diagnosis:notifications')


@login_required
def delete_notification(request, notification_id):
    notification = get_object_or_404(Notification, id=notification_id, user=request.user)
    notification.delete()
    if request.headers.get('x-requested-with') == 'XMLHttpRequest':
        return JsonResponse({'status': 'success'})
    messages.success(request, 'Notification deleted.')
    return redirect('Diagnosis:notifications')


# ============================================
# USER FEEDBACK VIEWS
# ============================================

@login_required
def feedback_view(request):
    profile = get_user_profile(request.user)
    feedbacks = UserFeedback.objects.filter(user=request.user).order_by('-created_at')
    avg_rating = feedbacks.aggregate(Avg('rating'))['rating__avg'] or 0
    total = feedbacks.count()
    rating_distribution = []
    for stars in range(5, 0, -1):
        count = feedbacks.filter(rating=stars).count()
        percentage = (count / total * 100) if total > 0 else 0
        rating_distribution.append({'stars': stars, 'count': count, 'percentage': round(percentage, 1)})

    context = {
        'user': request.user, 'profile': profile,
        'role': profile.get_role_display(),
        'subscription': profile.subscription_plan,
        'is_subscribed': profile.is_subscribed,
        'feedbacks': feedbacks,
        'avg_rating': round(avg_rating, 1),
        'rating_distribution': rating_distribution,
        'total_feedback': total,
    }
    return render(request, 'Diagnosis/feedback.html', context)


@login_required
def submit_feedback(request):
    if request.method == 'POST':
        rating = request.POST.get('rating')
        comment = request.POST.get('comment', '').strip()
        diagnosis_id = request.POST.get('diagnosis_id')

        if not rating:
            messages.error(request, 'Please select a rating.')
            return redirect('Diagnosis:feedback')

        try:
            rating = int(rating)
            if rating < 1 or rating > 5: raise ValueError
        except ValueError:
            messages.error(request, 'Invalid rating value.')
            return redirect('Diagnosis:feedback')

        feedback_data = {'user': request.user, 'rating': rating, 'comment': comment}
        if diagnosis_id:
            try:
                diagnosis = DiagnosisRecord.objects.get(id=diagnosis_id, user=request.user)
                feedback_data['diagnosis_record'] = diagnosis
            except DiagnosisRecord.DoesNotExist:
                pass

        UserFeedback.objects.create(**feedback_data)
        messages.success(request, 'Thank you for your feedback!')
        return redirect('Diagnosis:feedback')
    return redirect('Diagnosis:feedback')


@login_required
def delete_feedback(request, feedback_id):
    feedback = get_object_or_404(UserFeedback, id=feedback_id, user=request.user)
    feedback.delete()
    messages.success(request, 'Feedback deleted.')
    return redirect('Diagnosis:feedback')


# ╔══════════════════════════════════════════════════════════════════════════════════╗
# ║  DIAGNOSE IMAGE VIEW — XAI REMOVED, CACI ADDED                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════════╝

@login_required
@csrf_exempt
@require_http_methods(["POST"])
def diagnose_image(request):
    """Process medical image using the selected disease model"""
    try:
        profile = get_user_profile(request.user)
        image_file = request.FILES.get('image')
        if not image_file:
            return JsonResponse({'status': 'error', 'message': 'No image provided'}, status=400)

        disease_key = request.POST.get('disease', '')
        if not disease_key or disease_key not in MODEL_PATHS:
            return JsonResponse({'status': 'error', 'message': 'Invalid disease selection'}, status=400)

        patient_id = request.POST.get('patient_id', f"CASE-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}")
        modality = request.POST.get('modality', 'auto')

        print(f"🔍 Starting diagnosis for disease: {disease_key}")

        img_tensor = preprocess_image(image_file, disease_key=disease_key)

        model = load_model(disease_key)
        model.eval()

        start_time = datetime.now()
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item() * 100
        end_time = datetime.now()
        inference_time_ms = int((end_time - start_time).total_seconds() * 1000)

        class_names = CLASS_NAMES.get(disease_key, [])
        diagnosis = class_names[predicted_class] if predicted_class < len(class_names) else 'Unknown'
        print(f"✅ Diagnosis complete: {diagnosis} with {confidence:.1f}% confidence")

        probs = probabilities[0].tolist()
        prob_list = []
        colors = ['#00e5b0', '#3a9bff', '#7c6aff', '#ffc947', '#ff5575']
        for i, (name, prob) in enumerate(zip(class_names, probs)):
            prob_list.append([name, round(prob * 100, 1), colors[i % len(colors)]])

        is_positive = diagnosis in POSITIVE_CLASSES.get(disease_key, [])

        # ────────────────────────────────────────────────────────────
        # CONFIDENCE-AWARE CLINICAL INTERPRETATION (replaces XAI)
        # ────────────────────────────────────────────────────────────
        print(f"📋 Preparing clinical interpretation data...")

        sorted_probs = sorted(prob_list, key=lambda x: x[1], reverse=True)[:3]

        if confidence >= 90:
            strength = "Strong positive finding"
            strength_key = "strong"
        elif confidence >= 75:
            strength = "Moderate finding"
            strength_key = "moderate"
        elif confidence >= 60:
            strength = "Weak finding — recommend human review"
            strength_key = "weak"
        else:
            strength = "Inconclusive — recommend human review"
            strength_key = "inconclusive"

        margin = (sorted_probs[0][1] - sorted_probs[1][1]) if len(sorted_probs) > 1 else 100.0
        if margin >= 50:
            margin_label = "clear margin"
        elif margin >= 20:
            margin_label = "moderate margin"
        else:
            margin_label = "narrow margin — consider differential"

        clinical_summary = {
            "primary": diagnosis,
            "confidence": round(confidence, 1),
            "strength": strength,
            "strength_key": strength_key,
            "differential": sorted_probs,
            "margin": round(margin, 1),
            "margin_label": margin_label,
            "is_positive": is_positive,
        }
        print(f"✅ Clinical summary ready: {clinical_summary['primary']} "
              f"({clinical_summary['confidence']}%, {strength_key})")

        # ────────────────────────────────────────────────────────────
        # CREATE RECORD
        # ────────────────────────────────────────────────────────────
        record = DiagnosisRecord.objects.create(
            user=request.user,
            case_id=patient_id,
            modality=modality,
            disease_category=disease_key,
            diagnosis=diagnosis,
            confidence=round(confidence, 1),
            is_positive=is_positive,
            status='Completed',
            report_generated=True,
            inference_time=str(inference_time_ms),
            probabilities=prob_list,
            description=f'Diagnosis for {disease_key} - {diagnosis}'
        )

        # Store clinical_summary if the model supports it
        try:
            if hasattr(record, 'clinical_summary'):
                record.clinical_summary = clinical_summary
                record.save(update_fields=['clinical_summary'])
                print(f"   💾 Clinical summary stored")
        except Exception as e:
            print(f"   ⚠️ Could not store clinical summary: {e}")

        profile.total_diagnoses += 1
        profile.total_reports += 1
        profile.save()

        if is_positive:
            notif_message = f'{diagnosis} detected with {round(confidence, 1)}% confidence.'
        else:
            notif_message = f'No disease detected. Confidence: {round(confidence, 1)}%.'

        create_notification(
            request.user, 'diagnosis_complete', 'Diagnosis Complete',
            f'Your {disease_key.replace("_", " ").title()} analysis is complete. {notif_message}',
            f'/reports/{record.id}/'
        )

        # ────────────────────────────────────────────────────────────
        # RESPONSE
        # ────────────────────────────────────────────────────────────
        response_data = {
            'status': 'success',
            'id': record.id,
            'case_id': record.case_id,
            'diagnosis': diagnosis,
            'confidence': round(confidence, 1),
            'is_positive': is_positive,
            'probabilities': prob_list,
            'differential': sorted_probs,
            'clinical_summary': clinical_summary,
            'strength': strength,
            'margin': round(margin, 1),
            'margin_label': margin_label,
            'report_url': f'/reports/{record.id}/',
            'inference_time': inference_time_ms,
        }

        print(f"📤 Response keys: {list(response_data.keys())}")
        print(f"   clinical_summary: ✅")
        print(f"   differential:     ✅")

        return JsonResponse(response_data)

    except FileNotFoundError as e:
        print(f"❌ Model not found: {e}")
        return JsonResponse({'status': 'error', 'message': f'Model not found: {str(e)}'}, status=404)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback; traceback.print_exc()
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)


# ============================================
# MODEL CONFIGURATION
# ============================================

MODEL_PATHS = {
    'brain_tumor': os.path.join(settings.BASE_DIR, 'models', 'brain_tumor_best_model.pth'),
    'bone_fracture': os.path.join(settings.BASE_DIR, 'models', 'bone_fracture_best_model.pth'),
    'tuberculosis': os.path.join(settings.BASE_DIR, 'models', 'tb_best_model.pth'),
    'diabetic_retinopathy': os.path.join(settings.BASE_DIR, 'models', 'DR_best_model.pth'),
    'skin_cancer': os.path.join(settings.BASE_DIR, 'models', 'skincancer_best_model.pth'),
    'pneumonia': os.path.join(settings.BASE_DIR, 'models', 'pneumonia_best_model.pth'),
    'fracatlas': os.path.join(settings.BASE_DIR, 'models', 'fracatlas_best_model.pth'),
}

CLASS_NAMES = {
    'brain_tumor': ['No Tumor', 'Glioma Tumor', 'Meningioma', 'Pituitary Tumor'],
    'bone_fracture': ['No Fracture', 'Fracture'],
    'tuberculosis': ['Normal', 'Tuberculosis'],
    'diabetic_retinopathy': ['No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferate_DR'],
    'skin_cancer': ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc'],
    'pneumonia': ['Normal', 'Pneumonia'],
    'fracatlas': ['No Fracture', 'Fracture'],
}

POSITIVE_CLASSES = {
    'brain_tumor': ['Glioma Tumor', 'Meningioma', 'Pituitary Tumor'],
    'bone_fracture': ['Fracture'],
    'tuberculosis': ['Tuberculosis'],
    'diabetic_retinopathy': ['Mild', 'Moderate', 'Severe', 'Proliferate_DR'],
    'skin_cancer': ['bcc', 'mel', 'akiec'],
    'pneumonia': ['Pneumonia'],
    'fracatlas': ['Fracture'],
}

NUM_CLASSES = {
    'brain_tumor': 4, 'bone_fracture': 2, 'tuberculosis': 2,
    'diabetic_retinopathy': 5, 'skin_cancer': 7,
    'pneumonia': 2, 'fracatlas': 2,
}

MODEL_HPARAMS = {
    'brain_tumor': {'d_model': 384, 'nhead': 6, 'n_layers': 4, 'dim_ffn': 1536},
    'bone_fracture': {'d_model': 384, 'nhead': 6, 'n_layers': 4, 'dim_ffn': 1536},
    'tuberculosis': {'d_model': 384, 'nhead': 6, 'n_layers': 4, 'dim_ffn': 1536},
    'diabetic_retinopathy': {'d_model': 384, 'nhead': 6, 'n_layers': 4, 'dim_ffn': 1536},
    'skin_cancer': {'d_model': 384, 'nhead': 6, 'n_layers': 4, 'dim_ffn': 1536},
    'pneumonia': {'d_model': 384, 'nhead': 6, 'n_layers': 4, 'dim_ffn': 1536},
    'fracatlas': {'d_model': 384, 'nhead': 6, 'n_layers': 4, 'dim_ffn': 1536},
}

_model_cache = {}


# ============================================
# MODEL CLASSES
# ============================================

class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__(); self.drop_prob = drop_prob
    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training: return x
        return x * (torch.rand((x.shape[0], 1, 1), dtype=torch.float32, device=x.device) * float(1.0 - self.drop_prob))


class WindowAttention(nn.Module):
    def __init__(self, d_model, nhead, window_size=7, dropout=0.1):
        super().__init__()
        self.nhead, self.window_size, self.head_dim = nhead, window_size, d_model // nhead
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.proj = nn.Linear(d_model, d_model)
        self.attn_drop, self.proj_drop, self.attn_weights = nn.Dropout(dropout), nn.Dropout(dropout), None
    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.reshape(B, H, W, C)
        pad_h, pad_w = (self.window_size - H % self.window_size) % self.window_size, (self.window_size - W % self.window_size) % self.window_size
        if pad_h > 0 or pad_w > 0: x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        Hp, Wp = x.shape[1], x.shape[2]
        x = x.reshape(B, Hp//self.window_size, self.window_size, Wp//self.window_size, self.window_size, C).permute(0,1,3,2,4,5).reshape(-1, self.window_size**2, C)
        qkv = self.qkv(x).reshape(-1, self.window_size**2, 3, self.nhead, self.head_dim).permute(2,0,3,1,4)
        attn = (qkv[0] @ qkv[1].transpose(-2,-1)) * self.scale
        attn = attn.softmax(dim=-1); self.attn_weights = attn.detach()
        x = self.proj_drop(self.proj((self.attn_drop(attn) @ qkv[2]).transpose(1,2).reshape(-1, self.window_size**2, C)))
        x = x.reshape(B, Hp//self.window_size, Wp//self.window_size, self.window_size, self.window_size, C).permute(0,1,3,2,4,5).reshape(B, Hp, Wp, C)
        return x[:, :H, :W, :].reshape(B, -1, C) if pad_h > 0 or pad_w > 0 else x.reshape(B, -1, C)


class TransformerBlockWithWindow(nn.Module):
    def __init__(self, d_model, nhead, ffn_dim, window_size=7, attn_drop=0.1, ffn_drop=0.1, drop_path=0.0):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.global_attn = nn.MultiheadAttention(d_model, nhead, dropout=attn_drop, batch_first=True)
        self.window_attn = WindowAttention(d_model, nhead, window_size, attn_drop)
        self.gate = nn.Sequential(nn.Linear(d_model*2, d_model), nn.Sigmoid())
        self.ffn = nn.Sequential(nn.Linear(d_model, ffn_dim), nn.GELU(), nn.Dropout(ffn_drop), nn.Linear(ffn_dim, d_model), nn.Dropout(ffn_drop))
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.attn_weights, self.window_attn_weights = None, None
    def forward(self, x, H, W):
        x_norm = self.norm1(x)
        global_out, global_attn = self.global_attn(x_norm, x_norm, x_norm, need_weights=True)
        self.attn_weights = global_attn.detach()
        window_out = self.window_attn(x_norm[:, 1:, :], H, W)
        self.window_attn_weights = self.window_attn.attn_weights
        gate = self.gate(torch.cat([global_out[:, 1:, :], window_out], dim=-1))
        combined = torch.cat([global_out[:, :1, :], gate * global_out[:, 1:, :] + (1-gate) * window_out], dim=1)
        x = x + self.drop_path(combined)
        return x + self.drop_path(self.ffn(self.norm2(x)))


class ImprovedBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = tv_models.convnext_base(weights=tv_models.ConvNeXt_Base_Weights.IMAGENET1K_V1)
        self.feature_extractor = self.backbone.features
    def forward(self, x):
        return self.feature_extractor(x)


class EnhancedHViT(nn.Module):
    def __init__(self, num_classes=5, img_size=224, d_model=384, nhead=6, n_layers=4, dim_ffn=1536):
        super().__init__()
        self.d_model, self.img_size, self.num_classes = d_model, img_size, num_classes
        self.backbone = ImprovedBackbone()
        with torch.no_grad():
            _, c, h, w = self.backbone(torch.zeros(1, 3, img_size, img_size)).shape
        self.spatial_h, self.spatial_w = h, w
        self.proj = nn.Sequential(
            nn.Conv2d(c, d_model, 1, bias=False), nn.BatchNorm2d(d_model), nn.GELU(),
            nn.Conv2d(d_model, d_model, 3, padding=1, bias=False), nn.BatchNorm2d(d_model), nn.GELU()
        )
        self.norm_proj = nn.LayerNorm(d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, h*w + 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        dpr = [x.item() for x in torch.linspace(0, 0.15, n_layers)]
        self.encoder = nn.ModuleList([
            TransformerBlockWithWindow(d_model, nhead, dim_ffn, min(7, h, w), drop_path=dpr[i])
            for i in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model//2), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(d_model//2, d_model//4), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(d_model//4, num_classes)
        )
    def forward(self, x):
        B = x.size(0)
        tokens = rearrange(self.proj(self.backbone(x)), 'b d h w -> b (h w) d')
        tokens = self.norm_proj(tokens)
        tokens = torch.cat([self.cls_token.expand(B,-1,-1), tokens], dim=1) + self.pos_embed
        for block in self.encoder:
            tokens = block(tokens, self.spatial_h, self.spatial_w)
        return self.head(self.norm(tokens)[:, 0])
    def get_attentions(self):
        g, w = [], []
        for b in self.encoder:
            if b.attn_weights is not None: g.append(b.attn_weights)
            if b.window_attn_weights is not None: w.append(b.window_attn_weights)
        return g, w
    def unfreeze_backbone_partial(self, n=3):
        for p in self.backbone.feature_extractor.parameters():
            p.requires_grad_(False)
        for child in list(self.backbone.feature_extractor.children())[-n:]:
            for p in child.parameters():
                p.requires_grad_(True)


# ============================================
# MODEL LOADING FUNCTIONS
# ============================================

def load_model(disease_key):
    if disease_key in _model_cache:
        return _model_cache[disease_key]

    model_path = MODEL_PATHS.get(disease_key)
    if not model_path or not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found for disease: {disease_key}")

    num_classes = NUM_CLASSES.get(disease_key, 2)
    hparams = MODEL_HPARAMS.get(disease_key, {})

    model = EnhancedHViT(
        num_classes=num_classes, img_size=224,
        d_model=hparams.get('d_model', 384),
        nhead=hparams.get('nhead', 6),
        n_layers=hparams.get('n_layers', 4),
        dim_ffn=hparams.get('dim_ffn', 1536)
    )

    try:
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        model_dict = model.state_dict()
        head_key = 'head.7.weight'
        if head_key in state_dict:
            checkpoint_classes = state_dict[head_key].shape[0]
            if checkpoint_classes != num_classes:
                state_dict.pop('head.7.weight', None)
                state_dict.pop('head.7.bias', None)
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)
        model.eval()
        _model_cache[disease_key] = model
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model {disease_key}: {str(e)}")


def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def preprocess_image(image_file, disease_key=None):
    """Preprocess uploaded image for model input using PILImage."""
    try:
        print(f"📸 Processing image: {getattr(image_file, 'name', 'Unknown')}")
        print(f"📸 Image type: {type(image_file)}")
        print(f"📸 Image size: {getattr(image_file, 'size', 'Unknown')}")
        if disease_key:
            print(f"📸 Disease: {disease_key}")

        if hasattr(image_file, 'seek'):
            image_file.seek(0)

        img = PILImage.open(image_file).convert('RGB')
        transform = get_transforms()
        img_tensor = transform(img).unsqueeze(0)

        print(f"✅ Image tensor shape: {img_tensor.shape}")
        return img_tensor
    except Exception as e:
        print(f"❌ Image preprocessing error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise ValueError(f"Image preprocessing failed: {str(e)}")


# ============================================
# GENERATE INTERPRETATION VIEW (4-section, CACI-aware)
# ============================================

@login_required
@csrf_exempt
@require_http_methods(["POST"])
def generate_interpretation(request):
    try:
        data = json.loads(request.body)
        diagnosis = data.get('diagnosis', '')
        confidence = data.get('confidence', 0)
        modality = data.get('modality', 'Not specified')
        case_id = data.get('case_id', 'Unknown')
        differential = data.get('differential', [])
        strength = data.get('strength', '')
        margin_label = data.get('margin_label', '')
        force_local = data.get('force_local', False)

        if not diagnosis:
            return JsonResponse({'status': 'error', 'message': 'Diagnosis is required'}, status=400)

        # Build differential string
        diff_str = "\n".join([
            f"  {i+1}. {d[0]}: {d[1]}%" for i, d in enumerate(differential)
        ]) if differential else "  Not available"

        # Get disease info for context
        disease_info = DISEASE_INFO.get(diagnosis, {})
        severity = disease_info.get('severity', 'Unknown')
        symptoms = disease_info.get('common_symptoms', 'Not documented')
        treatment = disease_info.get('standard_treatment', 'Clinical correlation recommended')
        description = disease_info.get('description', 'No description available.')

        # Confidence band
        if confidence >= 90:
            conf_band = "very high"
        elif confidence >= 75:
            conf_band = "moderate-to-high"
        elif confidence >= 60:
            conf_band = "moderate"
        else:
            conf_band = "low"

        if force_local:
            return JsonResponse({
                'status': 'success',
                'interpretation': generate_local_interpretation_html(
                    diagnosis, confidence, modality, case_id,
                    differential=differential, strength=strength,
                    margin_label=margin_label
                ),
                'source': 'local'
            })

        groq_api_key = os.getenv('GROQ_API_KEY')
        print(f"🔑 GROQ_API_KEY exists: {bool(groq_api_key)}")

        if not groq_api_key:
            print("❌ GROQ_API_KEY not found in environment")
            return JsonResponse({
                'status': 'success',
                'interpretation': generate_local_interpretation_html(
                    diagnosis, confidence, modality, case_id,
                    differential=differential, strength=strength,
                    margin_label=margin_label
                ),
                'source': 'local_no_key'
            })

        print(f"📤 Sending detailed interpretation request to Groq for case: {case_id}")

        response = requests.post(
            'https://api.groq.com/openai/v1/chat/completions',
            headers={
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {groq_api_key}'
            },
            json={
                'model': 'openai/gpt-oss-120b',
                'messages': [
                    {
                        'role': 'system',
                        'content': """You are a senior clinical AI assistant generating detailed, structured diagnostic interpretations for radiologists and physicians.

OBJECTIVE:
Produce a comprehensive, clinically sound interpretation report that helps a physician understand:
- What the AI found
- How confident the finding is
- What alternatives were considered
- What the clinical context is
- What the physician should do next

STYLE RULES:
1. Write professionally but clearly, as a senior radiologist would in a structured report.
2. Use exactly 6 sections with these exact headings and numbering:
   - 1. PRIMARY FINDING
   - 2. DIFFERENTIAL DIAGNOSIS
   - 3. CLINICAL CONTEXT
   - 4. INTERPRETATION & CONFIDENCE
   - 5. RECOMMENDED NEXT STEPS
   - 6. CLINICAL RED FLAGS

3. Each section must have 4 to 5 bullet points.
4. Each bullet point must start with a dash "-" followed by a space.
5. Each bullet point must be 2 to 3 complete sentences. Do NOT write fragments.
6. Reference the confidence percentage, decision margin, and differential where clinically relevant.
7. Use plain medical English — no asterisks, no markdown, no slashes, no special symbols except dashes and periods.
8. Do not fabricate specific patient history you don't have — speak in probabilities and clinical reasoning.
9. Total output should be approximately 800 to 1200 words.
10. End the report with a one-line disclaimer on its own as the final bullet of section 6."""
                    },
                    {
                        'role': 'user',
                        'content': f"""Generate a detailed, structured clinical diagnostic interpretation report.

=== CASE INFORMATION ===
Case ID: {case_id}
Imaging Modality: {modality}
Primary AI Diagnosis: {diagnosis}
Confidence Score: {confidence}% ({conf_band} confidence)
Finding Strength: {strength}
Decision Margin: {margin_label}
Disease Severity: {severity}

=== TOP DIFFERENTIAL (from AI softmax) ===
{diff_str}

=== REFERENCE KNOWLEDGE (from clinical database) ===
Description: {description}
Common Symptoms: {symptoms}
Standard Treatment: {treatment}

=== REQUIRED OUTPUT FORMAT ===

1. PRIMARY FINDING
- Bullet 1 (2-3 sentences, name the finding, describe what it represents clinically, and state the confidence).
- Bullet 2 (2-3 sentences, explain the imaging modality used and what it typically reveals for this condition).
- Bullet 3 (2-3 sentences, describe the typical radiological appearance of this finding on {modality}).
- Bullet 4 (2-3 sentences, comment on severity — reference severity level "{severity}" and what that implies).
- Bullet 5 (2-3 sentences, summarize what the primary finding means for the patient at a high level).

2. DIFFERENTIAL DIAGNOSIS
- Bullet 1 (2-3 sentences, describe the top alternative class and why it remains plausible).
- Bullet 2 (2-3 sentences, describe the second alternative and its clinical plausibility).
- Bullet 3 (2-3 sentences, describe the third alternative and why it is less likely).
- Bullet 4 (2-3 sentences, explain what the decision margin "{margin_label}" tells us about differentiation).
- Bullet 5 (2-3 sentences, explain how a clinician would rule alternatives in or out).

3. CLINICAL CONTEXT
- Bullet 1 (2-3 sentences, describe the typical patient population affected by {diagnosis}).
- Bullet 2 (2-3 sentences, describe the common presenting symptoms: {symptoms}).
- Bullet 3 (2-3 sentences, describe the natural progression if untreated).
- Bullet 4 (2-3 sentences, describe risk factors and comorbidities often associated).
- Bullet 5 (2-3 sentences, describe how {modality} imaging usually fits into the diagnostic pathway).

4. INTERPRETATION & CONFIDENCE
- Bullet 1 (2-3 sentences, explain what a {confidence}% confidence actually means practically).
- Bullet 2 (2-3 sentences, explain what the finding strength "{strength}" implies for clinical action).
- Bullet 3 (2-3 sentences, explain how the decision margin "{margin_label}" affects reliability).
- Bullet 4 (2-3 sentences, explain what could cause false positives for this class).
- Bullet 5 (2-3 sentences, explain what could cause false negatives for this class).

5. RECOMMENDED NEXT STEPS
- Bullet 1 (2-3 sentences, describe immediate clinical action based on severity "{severity}").
- Bullet 2 (2-3 sentences, describe confirmatory tests or imaging follow-ups).
- Bullet 3 (2-3 sentences, describe specialist referral recommendation).
- Bullet 4 (2-3 sentences, describe short-term monitoring or follow-up schedule).
- Bullet 5 (2-3 sentences, describe standard treatment pathway: {treatment}).

6. CLINICAL RED FLAGS
- Bullet 1 (2-3 sentences, list acute symptoms that require emergency care).
- Bullet 2 (2-3 sentences, describe signs of rapid progression to watch for).
- Bullet 3 (2-3 sentences, describe complications specific to {diagnosis}).
- Bullet 4 (2-3 sentences, describe when to escalate to specialist or ER).
- Bullet 5 (1-2 sentences, mandatory disclaimer: this is AI-assisted interpretation and requires validation by a qualified clinician before any clinical decision).

=== FINAL RULES ===
- Write 4 to 5 bullets per section (25 to 30 bullets total).
- Each bullet is 2 to 3 sentences — no fragments.
- No asterisks, no slashes, no markdown, no hyphens except the leading dash.
- Use the exact section headings shown above.
- The report must be self-contained and readable without external context."""
                    }
                ],
                'temperature': 0.35,
                'max_tokens': 2500,
                'top_p': 0.9
            },
            timeout=60
        )

        print(f"📥 Response status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            interpretation = result.get('choices', [{}])[0].get('message', {}).get('content', '')
            if interpretation:
                print(f"✅ Success! Interpretation length: {len(interpretation)} chars")
                return JsonResponse({'status': 'success', 'interpretation': interpretation, 'source': 'groq'})

        print(f"❌ API request failed. Status: {response.status_code}")
        return JsonResponse({
            'status': 'success',
            'interpretation': generate_local_interpretation_html(
                diagnosis, confidence, modality, case_id,
                differential=differential, strength=strength,
                margin_label=margin_label
            ),
            'source': 'local_api_error'
        })

    except Exception as e:
        print(f"❌ Exception: {str(e)}")
        import traceback
        traceback.print_exc()
        return JsonResponse({
            'status': 'success',
            'interpretation': generate_local_interpretation_html(
                diagnosis, confidence, modality, case_id
            ),
            'source': 'local_error'
        })

def generate_local_interpretation_html(
    diagnosis, confidence, modality, case_id,
    differential=None, strength=None, margin_label=None
):
    """Fallback interpretation with 6 sections matching Groq output structure."""

    info = DISEASE_INFO.get(diagnosis, {
        'description': 'Condition detected requiring clinical correlation.',
        'severity': 'Unknown',
        'common_symptoms': 'Varies based on clinical presentation',
        'standard_treatment': 'Clinical correlation recommended'
    })

    severity = info.get('severity', 'Unknown')
    symptoms = info.get('common_symptoms', 'Not documented')
    treatment = info.get('standard_treatment', 'Clinical correlation recommended')
    description = info.get('description', 'No description available.')

    if confidence >= 90:
        conf_level = "High"
        conf_band = "very high"
    elif confidence >= 75:
        conf_level = "Moderate"
        conf_band = "moderate-to-high"
    elif confidence >= 60:
        conf_level = "Fair"
        conf_band = "moderate"
    else:
        conf_level = "Low"
        conf_band = "low"

    # Build differential HTML
    if differential:
        diff_html = "".join([
            f"<li><strong>{d[0]}:</strong> {d[1]}% probability. This class remains a plausible alternative "
            f"based on the model's softmax distribution and must be considered in the differential.</li>"
            for d in differential
        ])
    else:
        diff_html = "<li>Differential data not available for this case.</li>"

    strength_line = strength or f"{conf_level} confidence"
    margin_line = margin_label or "Not evaluated"

    # Margin interpretation
    if margin_label and "clear" in margin_label.lower():
        margin_note = "The clear decision margin indicates the model strongly favors this class over all alternatives."
    elif margin_label and "moderate" in margin_label.lower():
        margin_note = "The moderate decision margin indicates reasonable separation between this class and the next best alternative."
    elif margin_label and "narrow" in margin_label.lower():
        margin_note = "The narrow decision margin indicates the model is not strongly confident between top classes and human review is advised."
    else:
        margin_note = "The decision margin was not evaluated for this case."

    return f'''
    <div class="llm-interpretation-container">
        <div class="llm-header">
            <span class="llm-icon">AI</span>
            <span class="llm-title">Clinical Interpretation</span>
            <span class="llm-badge">Structured Report</span>
        </div>
        <div class="llm-body">
            <div class="llm-case-info">
                <div class="case-info-item"><span class="case-info-label">Case ID:</span><span class="case-info-value">{case_id}</span></div>
                <div class="case-info-item"><span class="case-info-label">Modality:</span><span class="case-info-value">{modality}</span></div>
                <div class="case-info-item"><span class="case-info-label">Diagnosis:</span><span class="case-info-value">{diagnosis}</span></div>
                <div class="case-info-item"><span class="case-info-label">Confidence:</span><span class="case-info-value">{confidence}% ({conf_level})</span></div>
                <div class="case-info-item"><span class="case-info-label">Strength:</span><span class="case-info-value">{strength_line}</span></div>
                <div class="case-info-item"><span class="case-info-label">Decision Margin:</span><span class="case-info-value">{margin_line}</span></div>
            </div>

            <div class="llm-numbered-section">
                <div class="llm-section-number">1. PRIMARY FINDING</div>
                <div class="llm-section-content"><ul>
                    <li>The primary AI finding is <strong>{diagnosis}</strong>, detected with a confidence of {confidence}% ({conf_band} confidence). This represents the model's strongest signal from the {modality} image analysis.</li>
                    <li>The {modality} modality was used for this analysis, which is standard for detecting and characterizing {diagnosis}. Imaging findings on {modality} provide the primary evidence base for this prediction.</li>
                    <li>{description}</li>
                    <li>The severity of this condition is classified as <strong>{severity}</strong>, which influences the urgency of clinical action. Higher severity findings typically require faster specialist involvement.</li>
                    <li>Overall, this primary finding should be treated as a strong candidate requiring clinical confirmation before any treatment decision.</li>
                </ul></div>
            </div>

            <div class="llm-numbered-section">
                <div class="llm-section-number">2. DIFFERENTIAL DIAGNOSIS</div>
                <div class="llm-section-content"><ul>
                    {diff_html}
                    <li>{margin_note}</li>
                    <li>A clinician should use targeted history, physical exam, and if needed follow-up imaging to rule in or rule out each differential class listed above.</li>
                </ul></div>
            </div>

            <div class="llm-numbered-section">
                <div class="llm-section-number">3. CLINICAL CONTEXT</div>
                <div class="llm-section-content"><ul>
                    <li>The typical patient population affected by {diagnosis} varies, but understanding demographics and comorbidities helps interpret this finding accurately.</li>
                    <li>Common presenting symptoms include: <strong>{symptoms}</strong>. These should be correlated with the patient's reported history.</li>
                    <li>If left untreated, this condition may progress and lead to complications depending on severity and patient factors.</li>
                    <li>Risk factors and comorbidities should be reviewed as part of the clinical workup to place this AI finding in proper context.</li>
                    <li>{modality} imaging typically serves as a key step in the diagnostic pathway, providing evidence that supports or refutes this AI prediction.</li>
                </ul></div>
            </div>

            <div class="llm-numbered-section">
                <div class="llm-section-number">4. INTERPRETATION &amp; CONFIDENCE</div>
                <div class="llm-section-content"><ul>
                    <li>A confidence of {confidence}% means the model is {conf_band} in this prediction. In practical terms, this affects how much weight the clinician should place on the AI output.</li>
                    <li>Finding strength is classified as <strong>{strength_line}</strong>, which guides the appropriate level of clinical response.</li>
                    <li>{margin_note}</li>
                    <li>False positives for this class can occur due to overlapping imaging features with other conditions, technical artifacts, or atypical presentations.</li>
                    <li>False negatives for this class can occur with subtle findings, early-stage disease, or limitations in image quality or resolution.</li>
                </ul></div>
            </div>

            <div class="llm-numbered-section">
                <div class="llm-section-number">5. RECOMMENDED NEXT STEPS</div>
                <div class="llm-section-content"><ul>
                    <li>Immediate clinical action should be tailored to the severity level <strong>{severity}</strong>. For high or critical severity, urgent specialist involvement is advised.</li>
                    <li>Confirmatory tests or imaging follow-up should be arranged to validate the AI finding, especially where confidence is below 90%.</li>
                    <li>Specialist referral is recommended based on the organ system and condition detected.</li>
                    <li>Short-term monitoring and follow-up scheduling should be planned to track progression or resolution.</li>
                    <li>Standard treatment pathway: <strong>{treatment}</strong>.</li>
                </ul></div>
            </div>

            <div class="llm-numbered-section">
                <div class="llm-section-number">6. CLINICAL RED FLAGS</div>
                <div class="llm-section-content"><ul>
                    <li>Acute symptoms requiring emergency care should be identified immediately, including severe pain, rapid deterioration, or life-threatening signs.</li>
                    <li>Signs of rapid progression include sudden worsening of symptoms, new neurological deficits, or systemic instability.</li>
                    <li>Complications specific to {diagnosis} should be monitored, especially those related to severity level <strong>{severity}</strong>.</li>
                    <li>Escalation to a specialist or emergency department is advised if red-flag symptoms appear or if the patient's condition deteriorates rapidly.</li>
                    <li>This is an AI-assisted interpretation and must be validated by a qualified clinician before any clinical decision is made.</li>
                </ul></div>
            </div>
        </div>
        <div class="llm-footer">Structured Clinical Interpretation — {datetime.now().strftime('%Y-%m-%d %H:%M')}</div>
    </div>
    '''

# ============================================
# SAVE HISTORY VIEW
# ============================================

@login_required
@csrf_exempt
@require_http_methods(["POST"])
def save_history(request):
    try:
        data = json.loads(request.body)
        record = DiagnosisRecord.objects.create(
            user=request.user,
            case_id=data.get('case_id', f"CASE-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"),
            modality=data.get('modality', 'Unknown'),
            disease_category=data.get('disease', ''),
            diagnosis=data.get('diagnosis', 'Unknown'),
            confidence=data.get('confidence', 0),
            is_positive=data.get('is_positive', False),
            status=data.get('status', 'Completed'),
            report_generated=data.get('report_generated', True),
            description=data.get('description', 'Saved from diagnosis')
        )
        profile = get_user_profile(request.user)
        profile.total_diagnoses += 1
        profile.total_reports += 1
        profile.save()
        return JsonResponse({'status': 'success', 'id': record.id, 'case_id': record.case_id})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)


# ============================================
# TEST API VIEWS
# ============================================

@login_required
def test_api(request):
    return JsonResponse({'status': 'success', 'message': 'API is working!', 'user': request.user.username})


@login_required
def test_model_loading(request):
    if not request.user.is_superuser:
        return JsonResponse({'status': 'error', 'message': 'Only superusers can test models'}, status=403)
    results = {}
    for disease_key, model_path in MODEL_PATHS.items():
        try:
            exists = os.path.exists(model_path)
            if exists:
                try:
                    model = load_model(disease_key)
                    results[disease_key] = {'path': model_path, 'exists': True, 'loaded': True, 'num_classes': NUM_CLASSES.get(disease_key, 0)}
                except Exception as e:
                    results[disease_key] = {'path': model_path, 'exists': True, 'loaded': False, 'error': str(e)}
            else:
                results[disease_key] = {'path': model_path, 'exists': False, 'loaded': False}
        except Exception as e:
            results[disease_key] = {'path': model_path, 'exists': os.path.exists(model_path), 'loaded': False, 'error': str(e)}
    return JsonResponse({'status': 'success', 'models': results})


# ============================================
# CREATE TEST RECORDS VIEW
# ============================================

@login_required
def create_test_records(request):
    if not request.user.is_superuser:
        return JsonResponse({'status': 'error', 'message': 'Only superusers can create test records'}, status=403)
    import random
    modalities = ['MRI', 'CT', 'X-ray', 'Micro']
    diagnoses = ['Glioma Tumor', 'Meningioma', 'Pituitary Tumor', 'Pneumonia', 'Tuberculosis', 'Melanoma', 'No Tumor', 'Fracture']
    statuses = ['Completed', 'Review', 'Completed', 'Completed', 'Review']
    created_count = 0
    for i in range(10):
        try:
            diagnosis = random.choice(diagnoses)
            is_positive = diagnosis not in ['No Tumor', 'Normal', 'No_DR']
            DiagnosisRecord.objects.create(
                user=request.user,
                case_id=f"TEST-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}",
                modality=random.choice(modalities),
                diagnosis=diagnosis,
                confidence=round(random.uniform(75, 99.5), 1),
                is_positive=is_positive,
                status=random.choice(statuses),
                report_generated=random.choice([True, False]),
                description=f"Test record {i+1} - Created for development/testing purposes"
            )
            created_count += 1
        except Exception as e:
            print(f"Error creating record {i+1}: {e}")
    return JsonResponse({'status': 'success', 'message': f'Created {created_count} test records successfully'})


# ============================================
# CHANGE PASSWORD VIEW
# ============================================

@login_required
def change_password(request):
    if request.method == 'POST':
        current_password = request.POST.get('current_password')
        new_password = request.POST.get('new_password')
        confirm_password = request.POST.get('confirm_password')
        errors = []
        if not request.user.check_password(current_password):
            errors.append('Current password is incorrect.')
        if len(new_password) < 8:
            errors.append('New password must be at least 8 characters.')
        if new_password != confirm_password:
            errors.append('Passwords do not match.')
        if errors:
            for error in errors: messages.error(request, error)
            return render(request, 'Diagnosis/change_password.html')
        request.user.set_password(new_password)
        request.user.save()
        from django.contrib.auth import update_session_auth_hash
        update_session_auth_hash(request, request.user)
        messages.success(request, 'Password changed successfully!')
        return redirect('Diagnosis:profile')
    return render(request, 'Diagnosis/change_password.html')


# ============================================
# UPLOAD AVATAR VIEW
# ============================================

@login_required
def upload_avatar(request):
    if request.method == 'POST' and request.FILES.get('profile_picture'):
        try:
            profile = request.user.profile
            profile.profile_picture = request.FILES['profile_picture']
            profile.save()
            return JsonResponse({'success': True, 'image_url': profile.profile_picture.url})
        except Exception as e:
            return JsonResponse({'success': False, 'message': str(e)})
    return JsonResponse({'success': False, 'message': 'No image provided'})


# ============================================
# API KEY STATUS TEST
# ============================================

@login_required
def test_api_key_status(request):
    from django.conf import settings
    groq_api_key = getattr(settings, 'GROQ_API_KEY', None)
    response_data = {
        'key_exists': bool(groq_api_key),
        'key_length': len(groq_api_key) if groq_api_key else 0,
        'key_prefix': groq_api_key[:10] if groq_api_key else 'None',
        'settings_module': settings.__module__,
    }
    if groq_api_key:
        try:
            test_response = requests.post(
                'https://api.groq.com/openai/v1/chat/completions',
                headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {groq_api_key}'},
                json={'model': 'llama-3.3-70b-versatile', 'messages': [{'role': 'user', 'content': 'Say "test successful"'}], 'max_tokens': 10},
                timeout=10
            )
            response_data['api_test'] = {
                'status_code': test_response.status_code,
                'success': test_response.status_code == 200,
                'response': test_response.json() if test_response.status_code == 200 else test_response.text[:200]
            }
        except Exception as e:
            response_data['api_test'] = {'error': str(e)}
    return JsonResponse(response_data)


# ============================================
# PASSWORD RESET VIEW
# ============================================

@csrf_exempt
def password_reset(request):
    if request.method != 'POST':
        return JsonResponse({'status': 'error', 'message': 'Method not allowed'}, status=405)
    try:
        data = json.loads(request.body)
        email = data.get('email', '').strip()
        if not email:
            return JsonResponse({'status': 'error', 'message': 'Email is required'}, status=400)
        try:
            users = User.objects.filter(email=email)
            if not users.exists():
                return JsonResponse({'status': 'success', 'message': 'If an account with this email exists, a reset link has been sent.'})
            user = users.first()
        except Exception as e:
            logger.error(f"Error finding user: {e}")
            return JsonResponse({'status': 'success', 'message': 'If an account with this email exists, a reset link has been sent.'})

        from django.contrib.auth.tokens import default_token_generator
        from django.utils.http import urlsafe_base64_encode
        from django.utils.encoding import force_bytes
        token = default_token_generator.make_token(user)
        uid = urlsafe_base64_encode(force_bytes(user.pk))
        reset_url = f"{request.scheme}://{request.get_host()}/reset-password/{uid}/{token}/"

        try:
            subject = 'Password Reset - X-HViT Medical'
            html_message = f"""
            <html>
            <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px; background: #0a0f1a; color: #e8edf5;">
                <div style="background: #111927; border-radius: 16px; padding: 30px; border: 1px solid rgba(255,255,255,0.06);">
                    <div style="text-align: center; margin-bottom: 24px;">
                        <div style="display: inline-block; background: linear-gradient(135deg, #00e5b0, #3a9bff); border-radius: 12px; padding: 8px 16px; font-size: 20px; font-weight: 700; color: #0a0f1a;">X-HViT</div>
                    </div>
                    <h2 style="color: #00e5b0; margin-bottom: 16px;">Password Reset Request</h2>
                    <p style="color: #8892a8; line-height: 1.6;">We received a request to reset your password for your X-HViT Medical account.</p>
                    <div style="text-align: center; margin: 24px 0;">
                        <a href="{reset_url}" style="display: inline-block; padding: 12px 32px; background: linear-gradient(135deg, #00e5b0, #00c49a); color: #0a0f1a; text-decoration: none; border-radius: 8px; font-weight: 600;">Reset Password</a>
                    </div>
                    <p style="color: #8892a8; line-height: 1.6;">If you didn't request this, please ignore this email. Your password will remain unchanged.</p>
                    <p style="color: #4a5568; font-size: 12px; margin-top: 24px; border-top: 1px solid rgba(255,255,255,0.06); padding-top: 16px;">This link will expire in 24 hours. For security, do not share this email with anyone.</p>
                </div>
            </body>
            </html>
            """
            plain_message = f"""Password Reset - X-HViT Medical

We received a request to reset your password.

Click the link below to reset your password:
{reset_url}

If you didn't request this, please ignore this email.

This link will expire in 24 hours."""
            send_mail(subject, plain_message, settings.DEFAULT_FROM_EMAIL, [email],
                      html_message=html_message, fail_silently=False)
            return JsonResponse({'status': 'success', 'message': 'Reset link sent successfully.'})
        except Exception as e:
            logger.error(f"Email sending error: {e}")
            return JsonResponse({'status': 'error', 'message': 'Failed to send email. Please try again.'}, status=500)
    except json.JSONDecodeError:
        return JsonResponse({'status': 'error', 'message': 'Invalid request data'}, status=400)
    except Exception as e:
        logger.error(f"Password reset error: {e}")
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)
    
@login_required
def explainability_view(request, report_id):
    """Explainability Hub — replaces XAI heatmaps with CACI reasoning."""
    try:
        profile = get_user_profile(request.user)
        record = get_object_or_404(DiagnosisRecord, id=report_id, user=request.user)

        disease_info = DISEASE_INFO.get(record.diagnosis, {})

        probabilities = []
        if record.probabilities:
            try:
                if isinstance(record.probabilities, list):
                    probabilities = record.probabilities
                elif isinstance(record.probabilities, str):
                    probabilities = json.loads(record.probabilities)
            except:
                probabilities = []

        if not probabilities:
            probabilities = [[record.diagnosis, record.confidence, '#00e5b0']]

        similar_cases = DiagnosisRecord.objects.filter(
            user=request.user,
            diagnosis=record.diagnosis
        ).exclude(id=record.id).order_by('-confidence')[:10]

        context = {
            'user': request.user,
            'profile': profile,
            'record': record,
            'disease_info': disease_info,
            'probabilities': probabilities,
            'similar_cases': similar_cases,
        }
        return render(request, 'Diagnosis/explainability.html', context)

    except DiagnosisRecord.DoesNotExist:
        messages.error(request, 'Report not found.')
        return redirect('Diagnosis:reports')