# Diagnosis/models.py - Complete file with aliases

from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
import uuid
from decimal import Decimal

# ============================================
# CORE MODELS
# ============================================


class DiagnosisRecord(models.Model):
    """Main model for storing diagnosis records"""
    
    MODALITY_CHOICES = [
        ('MRI', 'MRI'),
        ('CT', 'CT'),
        ('X-ray', 'X-ray'),
        ('Micro', 'Micro'),
        ('auto', 'Auto'),
    ]
    
    STATUS_CHOICES = [
        ('Completed', 'Completed'),
        ('Review', 'Review'),
        ('Processing', 'Processing'),
        ('Failed', 'Failed'),
    ]
    
    XAI_MODE_CHOICES = [
        ('both', 'Grad-CAM + Attention'),
        ('gradcam', 'Grad-CAM Only'),
        ('attention', 'Attention Only'),
    ]
    
    # Auto-generate case ID if not provided
    def generate_case_id(self):
        if self.pk and self.created_at:
            return f"CASE-{self.created_at.strftime('%Y')}-{uuid.uuid4().hex[:6].upper()}"
        return f"CASE-{uuid.uuid4().hex[:8].upper()}"
    
    # User relationship
    user = models.ForeignKey(
        User, 
        on_delete=models.CASCADE, 
        related_name='diagnosis_records'
    )
    
    # Case information
    case_id = models.CharField(max_length=50, unique=True, blank=True)
    modality = models.CharField(max_length=20, choices=MODALITY_CHOICES, blank=True, null=True)
    disease_category = models.CharField(max_length=50, blank=True, null=True)
    xai_mode = models.CharField(max_length=20, choices=XAI_MODE_CHOICES, default='both')
    
    # Diagnosis results
    diagnosis = models.CharField(max_length=200)
    confidence = models.FloatField(default=0.0)
    probabilities = models.JSONField(default=list, blank=True, null=True)
    is_positive = models.BooleanField(default=False)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='Completed')
    
    # Files and paths - UPDATED to use TextField for base64 data
    report_generated = models.BooleanField(default=False)
    description = models.TextField(blank=True, null=True)
    image_path = models.CharField(max_length=255, blank=True, null=True)
    
    # ✅ Changed to TextField to store base64 data (which can be very long)
    gradcam_path = models.TextField(blank=True, null=True)  # Was CharField(max_length=255)
    attention_path = models.TextField(blank=True, null=True)  # Was CharField(max_length=255)
    combined_path = models.TextField(blank=True, null=True) 
    panel_path = models.TextField(blank=True, null=True)
    transformer_path = models.TextField(blank=True, null=True)
    
    report_path = models.CharField(max_length=255, blank=True, null=True)
    
    # Additional metadata
    inference_time = models.CharField(max_length=50, blank=True, null=True)
    download_count = models.IntegerField(default=0)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def save(self, *args, **kwargs):
        if not self.case_id:
            self.case_id = self.generate_case_id()
        super().save(*args, **kwargs)
    
    def __str__(self):
        return f"{self.case_id} - {self.diagnosis} ({self.user.username})"
    
    def get_modality_display(self):
        """Get human-readable modality name"""
        return dict(self.MODALITY_CHOICES).get(self.modality, self.modality or 'Unknown')
    
    def get_status_display(self):
        """Get human-readable status name"""
        return dict(self.STATUS_CHOICES).get(self.status, self.status or 'Unknown')
    
    def get_xai_mode_display(self):
        """Get human-readable XAI mode name"""
        return dict(self.XAI_MODE_CHOICES).get(self.xai_mode, self.xai_mode or 'Both')
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Diagnosis Record'
        verbose_name_plural = 'Diagnosis Records'
        

class ContactMessage(models.Model):
    """Model to store contact form submissions"""
    
    CATEGORY_CHOICES = [
        ('bug', 'Bug Report'),
        ('feature', 'Feature Request'),
        ('research', 'Research Collaboration'),
        ('general', 'General Question'),
        ('payment', 'Payment Inquiry'),
        ('other', 'Other'),
    ]
    
    STATUS_CHOICES = [
        ('new', 'New'),
        ('read', 'Read'),
        ('replied', 'Replied'),
        ('resolved', 'Resolved'),
    ]
    
    user = models.ForeignKey(
        User, 
        on_delete=models.SET_NULL, 
        null=True, 
        blank=True, 
        related_name='contact_messages'
    )
    first_name = models.CharField(max_length=100)
    last_name = models.CharField(max_length=100)
    email = models.EmailField()
    category = models.CharField(max_length=50, choices=CATEGORY_CHOICES, default='general')
    subject = models.CharField(max_length=200)
    message = models.TextField()
    status = models.CharField(max_length=50, choices=STATUS_CHOICES, default='new')
    admin_notes = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Contact Message'
        verbose_name_plural = 'Contact Messages'
    
    def __str__(self):
        return f"{self.subject} - {self.email} ({self.created_at.strftime('%Y-%m-%d')})"
    
    def get_full_name(self):
        return f"{self.first_name} {self.last_name}".strip()


# ============================================
# USER PROFILE & SUBSCRIPTION
# ============================================

class UserProfile(models.Model):
    """Extended user profile information"""
    
    ROLE_CHOICES = [
        ('researcher', 'Researcher'),
        ('clinician', 'Clinician'),
        ('student', 'Student'),
        ('admin', 'Administrator'),
    ]
    
    user = models.OneToOneField(
        User, 
        on_delete=models.CASCADE, 
        related_name='profile'
    )
    role = models.CharField(max_length=50, choices=ROLE_CHOICES, default='researcher')
    institution = models.CharField(max_length=200, blank=True, null=True)
    phone = models.CharField(max_length=20, blank=True, null=True)
    profile_picture = models.ImageField(upload_to='profiles/', blank=True, null=True)
    
    # Subscription
    subscription_plan = models.CharField(max_length=50, default='free')
    subscription_expiry = models.DateTimeField(null=True, blank=True)
    is_subscribed = models.BooleanField(default=False)
    
    # Stats
    total_diagnoses = models.IntegerField(default=0)
    total_reports = models.IntegerField(default=0)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.user.username} - {self.role}"
    
    def get_full_name(self):
        return self.user.get_full_name() or self.user.username
    
    def has_active_subscription(self):
        """Check if user has an active subscription"""
        if not self.is_subscribed:
            return False
        if self.subscription_expiry:
            return timezone.now() < self.subscription_expiry
        return True
    
    class Meta:
        verbose_name = 'User Profile'
        verbose_name_plural = 'User Profiles'


class SubscriptionPlan(models.Model):
    """Pricing plans"""
    
    PLAN_TYPES = [
        ('free', 'Free'),
        ('research', 'Research'),
        ('clinical', 'Clinical'),
        ('enterprise', 'Enterprise'),
    ]
    
    name = models.CharField(max_length=100)
    plan_type = models.CharField(max_length=50, choices=PLAN_TYPES, unique=True)
    price_per_month = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    price_per_year = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    diagnoses_limit = models.IntegerField(default=10)  # -1 for unlimited
    features = models.JSONField(default=list)  # List of features
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.name} - NPRs. {self.price_per_month}/mo"
    
    def get_features_list(self):
        """Return features as a list"""
        if isinstance(self.features, list):
            return self.features
        return []
    
    class Meta:
        verbose_name = 'Subscription Plan'
        verbose_name_plural = 'Subscription Plans'


class PaymentTransaction(models.Model):
    """Track payment transactions"""
    
    PAYMENT_STATUS = [
        ('pending', 'Pending'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
        ('refunded', 'Refunded'),
    ]
    
    PAYMENT_METHOD = [
        ('esewa', 'eSewa'),
        ('khalti', 'Khalti'),
        ('bank', 'Bank Transfer'),
        ('card', 'Credit Card'),
    ]
    
    user = models.ForeignKey(
        User, 
        on_delete=models.CASCADE, 
        related_name='payments'
    )
    plan = models.ForeignKey(
        SubscriptionPlan, 
        on_delete=models.SET_NULL, 
        null=True, 
        related_name='payments'
    )
    transaction_id = models.CharField(max_length=100, unique=True)
    amount = models.DecimalField(max_digits=10, decimal_places=2)
    payment_method = models.CharField(max_length=50, choices=PAYMENT_METHOD, default='esewa')
    status = models.CharField(max_length=50, choices=PAYMENT_STATUS, default='pending')
    receipt_url = models.CharField(max_length=500, blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    
    def __str__(self):
        return f"{self.transaction_id} - {self.user.username} - {self.status}"
    
    def mark_completed(self):
        """Mark transaction as completed"""
        self.status = 'completed'
        self.completed_at = timezone.now()
        self.save()
        
        # Update user subscription
        if self.plan:
            profile = self.user.profile
            profile.subscription_plan = self.plan.plan_type
            profile.is_subscribed = True
            profile.subscription_expiry = timezone.now() + timezone.timedelta(days=30)
            profile.save()
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Payment Transaction'
        verbose_name_plural = 'Payment Transactions'


# ============================================
# AI & MODELS
# ============================================

class AIModelInfo(models.Model):
    """Store information about available AI models"""
    
    MODEL_TYPES = [
        ('brain_tumor', 'Brain Tumor'),
        ('bone_fracture', 'Bone Fracture'),
        ('tuberculosis', 'Tuberculosis'),
        ('diabetic_retinopathy', 'Diabetic Retinopathy'),
        ('skin_cancer', 'Skin Cancer'),
        ('pneumonia', 'Pneumonia'),
        ('fracatlas', 'FracAtlas'),
        ('custom', 'Custom Model'),
    ]
    
    name = models.CharField(max_length=100)
    model_type = models.CharField(max_length=50, choices=MODEL_TYPES, unique=True)
    version = models.CharField(max_length=20, default='1.0.0')
    description = models.TextField()
    accuracy = models.FloatField(null=True, blank=True)
    num_classes = models.IntegerField(default=2)
    class_names = models.JSONField(default=list)
    model_path = models.CharField(max_length=500)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.name} v{self.version}"
    
    def get_accuracy_percentage(self):
        """Return accuracy as percentage"""
        if self.accuracy:
            return f"{self.accuracy * 100:.1f}%"
        return 'N/A'
    
    class Meta:
        verbose_name = 'AI Model Info'
        verbose_name_plural = 'AI Model Infos'


# ============================================
# ACTIVITY LOG
# ============================================

class ActivityLog(models.Model):
    """Log user activities for audit and analytics"""
    
    ACTION_TYPES = [
        ('login', 'Login'),
        ('logout', 'Logout'),
        ('diagnosis', 'Diagnosis'),
        ('report', 'Report Generated'),
        ('download', 'Download'),
        ('payment', 'Payment'),
        ('profile_update', 'Profile Update'),
        ('feedback', 'Feedback Submitted'),
        ('contact', 'Contact Form'),
        ('view', 'Page View'),
        ('other', 'Other'),
    ]
    
    user = models.ForeignKey(
        User, 
        on_delete=models.CASCADE, 
        related_name='activities'
    )
    action_type = models.CharField(max_length=50, choices=ACTION_TYPES)
    description = models.TextField()
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    user_agent = models.TextField(blank=True, null=True)
    
    # Optional: Link to related objects
    related_record_id = models.IntegerField(null=True, blank=True, 
        help_text="ID of related record (e.g., DiagnosisRecord ID)")
    related_model = models.CharField(max_length=100, blank=True, null=True,
        help_text="Name of related model")
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Activity Log'
        verbose_name_plural = 'Activity Logs'
    
    def __str__(self):
        return f"{self.user.username} - {self.action_type} - {self.created_at.strftime('%Y-%m-%d %H:%M')}"
    
    def get_action_icon(self):
        """Return icon for action type"""
        icons = {
            'login': '🔐',
            'logout': '🚪',
            'diagnosis': '🧠',
            'report': '📄',
            'download': '⬇️',
            'payment': '💰',
            'profile_update': '👤',
            'feedback': '⭐',
            'contact': '📧',
            'view': '👁️',
        }
        return icons.get(self.action_type, '📋')


class Notification(models.Model):
    """User notifications"""
    
    NOTIFICATION_TYPES = [
        ('diagnosis_complete', 'Diagnosis Complete'),
        ('report_ready', 'Report Ready'),
        ('payment_success', 'Payment Success'),
        ('subscription_expiry', 'Subscription Expiry'),
        ('welcome', 'Welcome'),
        ('feedback_request', 'Feedback Request'),
        ('system_update', 'System Update'),
        ('other', 'Other'),
    ]
    
    user = models.ForeignKey(
        User, 
        on_delete=models.CASCADE, 
        related_name='notifications'
    )
    notification_type = models.CharField(max_length=50, choices=NOTIFICATION_TYPES)
    title = models.CharField(max_length=200)
    message = models.TextField()
    link = models.CharField(max_length=500, blank=True, null=True)
    is_read = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Notification'
        verbose_name_plural = 'Notifications'
    
    def __str__(self):
        return f"{self.user.username} - {self.title}"
    
    def mark_as_read(self):
        self.is_read = True
        self.save()


# ============================================
# FEEDBACK & SAVED ANALYSES
# ============================================

class UserFeedback(models.Model):
    """User feedback and ratings for diagnoses"""
    
    RATING_CHOICES = [(i, str(i)) for i in range(1, 6)]
    
    user = models.ForeignKey(
        User, 
        on_delete=models.CASCADE, 
        related_name='feedback'
    )
    diagnosis_record = models.ForeignKey(
        DiagnosisRecord, 
        on_delete=models.SET_NULL, 
        null=True, 
        blank=True, 
        related_name='feedback'
    )
    rating = models.IntegerField(choices=RATING_CHOICES)
    comment = models.TextField(blank=True, null=True)
    is_helpful = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.user.username} - Rating: {self.rating}/5"
    
    def get_rating_stars(self):
        """Return star representation of rating"""
        return '⭐' * self.rating
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = 'User Feedback'
        verbose_name_plural = 'User Feedbacks'


class SavedAnalysis(models.Model):
    """User saved analyses for later reference"""
    
    user = models.ForeignKey(
        User, 
        on_delete=models.CASCADE, 
        related_name='saved_analyses'
    )
    diagnosis_record = models.ForeignKey(
        DiagnosisRecord, 
        on_delete=models.CASCADE, 
        related_name='saved_by'
    )
    note = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
        unique_together = ['user', 'diagnosis_record']
        verbose_name = 'Saved Analysis'
        verbose_name_plural = 'Saved Analyses'
    
    def __str__(self):
        return f"{self.user.username} - {self.diagnosis_record.case_id}"


# ============================================
# SYSTEM SETTINGS
# ============================================

class SystemSetting(models.Model):
    """System-wide settings and configurations"""
    
    SETTING_TYPES = [
        ('general', 'General'),
        ('ai', 'AI Settings'),
        ('security', 'Security'),
        ('email', 'Email Settings'),
        ('payment', 'Payment Settings'),
    ]
    
    key = models.CharField(max_length=100, unique=True)
    value = models.TextField()
    setting_type = models.CharField(max_length=50, choices=SETTING_TYPES, default='general')
    description = models.TextField(blank=True, null=True)
    is_public = models.BooleanField(default=False)  # Exposed to frontend
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.key} - {self.setting_type}"
    
    def get_value_as_bool(self):
        """Return value as boolean"""
        return self.value.lower() in ('true', '1', 'yes', 'on')
    
    def get_value_as_int(self):
        """Return value as integer"""
        try:
            return int(self.value)
        except (ValueError, TypeError):
            return None
    
    class Meta:
        verbose_name = 'System Setting'
        verbose_name_plural = 'System Settings'


# ============================================
# BACKWARD COMPATIBILITY ALIASES
# ============================================

class DiagnosisReport(DiagnosisRecord):
    """
    Alias for DiagnosisRecord - maintains backward compatibility
    with code that expects a DiagnosisReport model.
    """
    class Meta:
        proxy = True
        verbose_name = 'Diagnosis Report'
        verbose_name_plural = 'Diagnosis Reports'


class DiagnosisHistory(DiagnosisRecord):
    """
    Alias for DiagnosisRecord - maintains backward compatibility
    with code that expects a DiagnosisHistory model.
    """
    class Meta:
        proxy = True
        verbose_name = 'Diagnosis History'
        verbose_name_plural = 'Diagnosis Histories'


class Modality(models.Model):
    """
    Modality model for imaging types.
    This is a separate model for backward compatibility.
    """
    MODALITY_TYPES = [
        ('MRI', 'MRI'),
        ('CT', 'CT'),
        ('X-ray', 'X-ray'),
        ('Micro', 'Micro'),
        ('auto', 'Auto'),
    ]
    
    name = models.CharField(max_length=50, choices=MODALITY_TYPES, unique=True)
    display_name = models.CharField(max_length=50)
    description = models.TextField(blank=True, null=True)
    icon = models.CharField(max_length=50, blank=True, null=True)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return self.display_name or self.name
    
    class Meta:
        verbose_name = 'Modality'
        verbose_name_plural = 'Modalities'


def get_modality_choices():
    """Return modality choices as a list of tuples"""
    return [
        ('MRI', 'MRI'),
        ('CT', 'CT'),
        ('X-ray', 'X-ray'),
        ('Micro', 'Micro'),
        ('auto', 'Auto'),
    ]


# ============================================
# HELPERS
# ============================================

def create_activity_log(user, action_type, description, request=None, related_id=None, related_model=None):
    """
    Helper to create activity logs
    
    Args:
        user: User instance
        action_type: Type of action (from ACTION_TYPES)
        description: Description of the action
        request: HTTP request (optional - for IP and user agent)
        related_id: ID of related record (optional)
        related_model: Name of related model (optional)
    
    Returns:
        ActivityLog instance
    """
    activity = ActivityLog(
        user=user,
        action_type=action_type,
        description=description,
        related_record_id=related_id,
        related_model=related_model
    )
    if request:
        activity.ip_address = request.META.get('REMOTE_ADDR')
        activity.user_agent = request.META.get('HTTP_USER_AGENT', '')
    activity.save()
    return activity


def create_notification(user, notification_type, title, message, link=None):
    """Helper to create notifications"""
    notification = Notification(
        user=user,
        notification_type=notification_type,
        title=title,
        message=message,
        link=link
    )
    notification.save()
    return notification


def get_user_diagnosis_stats(user):
    """Get diagnosis statistics for a user"""
    records = DiagnosisRecord.objects.filter(user=user)
    
    stats = {
        'total': records.count(),
        'completed': records.filter(status='Completed').count(),
        'review': records.filter(status='Review').count(),
        'failed': records.filter(status='Failed').count(),
        'avg_confidence': records.aggregate(models.Avg('confidence'))['confidence__avg'] or 0,
        'positive_count': records.filter(is_positive=True).count(),
        'report_count': records.filter(report_generated=True).count(),
    }
    
    # Modality breakdown
    modality_counts = {}
    for record in records:
        modality = record.modality or 'Unknown'
        modality_counts[modality] = modality_counts.get(modality, 0) + 1
    
    stats['modality_counts'] = modality_counts
    
    return stats