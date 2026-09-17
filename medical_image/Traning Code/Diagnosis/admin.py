# Diagnosis/admin.py - FIXED

from django.contrib import admin
from django.contrib.auth.models import User
from django.contrib.auth.admin import UserAdmin
from django.utils.html import format_html
from django.utils import timezone
from .models import (
    DiagnosisRecord,
    ContactMessage,
    UserProfile,
    SubscriptionPlan,
    PaymentTransaction,
    AIModelInfo,
    ActivityLog,
    Notification,
    UserFeedback,
    SavedAnalysis,
    SystemSetting
)


# ============================================
# CUSTOM ADMIN SITE
# ============================================
class CustomAdminSite(admin.AdminSite):
    site_header = "X-HViT Administration"
    site_title = "X-HViT Admin"
    index_title = "Dashboard - X-HViT Medical AI"


# ============================================
# DIAGNOSIS RECORD ADMIN
# ============================================
@admin.register(DiagnosisRecord)
class DiagnosisRecordAdmin(admin.ModelAdmin):
    list_display = (
        'case_id', 
        'user', 
        'diagnosis', 
        'confidence', 
        'modality', 
        'is_positive',
        'status', 
        'created_at'
    )
    list_filter = (
        'modality', 
        'status', 
        'is_positive', 
        'report_generated',
        'created_at'
    )
    search_fields = (
        'case_id', 
        'diagnosis', 
        'user__username', 
        'user__email',
        'description'
    )
    readonly_fields = (
        'case_id', 
        'created_at', 
        'updated_at',
        'download_count'
    )
    list_per_page = 50
    date_hierarchy = 'created_at'
    ordering = ['-created_at']
    
    fieldsets = (
        ('Case Information', {
            'fields': (
                'case_id', 
                'user', 
                'modality', 
                'disease_category',
                'xai_mode'
            )
        }),
        ('Diagnosis Results', {
            'fields': (
                'diagnosis', 
                'confidence', 
                'probabilities', 
                'is_positive',
                'status'
            )
        }),
        ('Files & Paths', {
            'fields': (
                'image_path', 
                'gradcam_path', 
                'attention_path', 
                'report_path'
            ),
            'classes': ('collapse',)
        }),
        ('Metadata', {
            'fields': (
                'report_generated', 
                'description', 
                'inference_time',
                'download_count'
            )
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )
    
    def get_queryset(self, request):
        qs = super().get_queryset(request)
        if request.user.is_superuser:
            return qs
        return qs.filter(user=request.user)
    
    def save_model(self, request, obj, form, change):
        if not obj.pk:
            obj.user = request.user
        super().save_model(request, obj, form, change)
    
    actions = ['mark_as_completed', 'mark_as_review', 'mark_as_failed']
    
    def mark_as_completed(self, request, queryset):
        queryset.update(status='Completed')
    mark_as_completed.short_description = "Mark selected as Completed"
    
    def mark_as_review(self, request, queryset):
        queryset.update(status='Review')
    mark_as_review.short_description = "Mark selected as Review"
    
    def mark_as_failed(self, request, queryset):
        queryset.update(status='Failed')
    mark_as_failed.short_description = "Mark selected as Failed"
    
    def get_actions(self, request):
        actions = super().get_actions(request)
        if 'delete_selected' in actions:
            del actions['delete_selected']
        return actions


# ============================================
# CONTACT MESSAGE ADMIN
# ============================================
@admin.register(ContactMessage)
class ContactMessageAdmin(admin.ModelAdmin):
    list_display = (
        'subject', 
        'get_full_name', 
        'email', 
        'category', 
        'status',
        'created_at'
    )
    list_filter = ('category', 'status', 'created_at')
    search_fields = ('first_name', 'last_name', 'email', 'subject', 'message')
    readonly_fields = ('created_at', 'updated_at')
    list_per_page = 30
    ordering = ['-created_at']
    
    fieldsets = (
        ('Contact Information', {
            'fields': (
                'first_name', 
                'last_name', 
                'email', 
                'category'
            )
        }),
        ('Message', {
            'fields': (
                'subject', 
                'message'
            )
        }),
        ('Status', {
            'fields': (
                'status', 
                'admin_notes'
            )
        }),
        ('Metadata', {
            'fields': (
                'user', 
                'created_at', 
                'updated_at'
            ),
            'classes': ('collapse',)
        }),
    )
    
    actions = ['mark_as_read', 'mark_as_replied', 'mark_as_resolved']
    
    def mark_as_read(self, request, queryset):
        queryset.update(status='read')
    mark_as_read.short_description = "Mark selected as Read"
    
    def mark_as_replied(self, request, queryset):
        queryset.update(status='replied')
    mark_as_replied.short_description = "Mark selected as Replied"
    
    def mark_as_resolved(self, request, queryset):
        queryset.update(status='resolved')
    mark_as_resolved.short_description = "Mark selected as Resolved"
    
    def get_full_name(self, obj):
        return obj.get_full_name()
    get_full_name.short_description = "Full Name"


# ============================================
# USER PROFILE ADMIN
# ============================================
@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = (
        'user', 
        'get_role_display', 
        'institution', 
        'subscription_plan',
        'is_subscribed',
        'total_diagnoses',
        'total_reports'
    )
    list_filter = (
        'role', 
        'subscription_plan', 
        'is_subscribed',
        'created_at'
    )
    search_fields = (
        'user__username', 
        'user__email', 
        'user__first_name', 
        'user__last_name',
        'institution', 
        'phone'
    )
    readonly_fields = (
        'created_at', 
        'updated_at',
        'total_diagnoses',
        'total_reports'
    )
    list_per_page = 30
    ordering = ['-created_at']
    
    fieldsets = (
        ('User Information', {
            'fields': (
                'user', 
                'role', 
                'institution', 
                'phone'
            )
        }),
        ('Subscription', {
            'fields': (
                'subscription_plan', 
                'subscription_expiry', 
                'is_subscribed'
            )
        }),
        ('Statistics', {
            'fields': (
                'total_diagnoses', 
                'total_reports'
            )
        }),
        ('Profile Picture', {
            'fields': ('profile_picture',),
            'classes': ('collapse',)
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )
    
    def get_role_display(self, obj):
        return obj.get_role_display()
    get_role_display.short_description = "Role"


# ============================================
# SUBSCRIPTION PLAN ADMIN - FIXED
# ============================================
@admin.register(SubscriptionPlan)
class SubscriptionPlanAdmin(admin.ModelAdmin):
    list_display = (
        'name', 
        'plan_type', 
        'price_per_month', 
        'price_per_year',
        'diagnoses_limit',
        'is_active'
    )
    list_filter = ('plan_type', 'is_active')
    search_fields = ('name', 'description')
    readonly_fields = ('created_at', 'updated_at')  # Added readonly fields
    list_per_page = 20
    
    fieldsets = (
        ('Plan Information', {
            'fields': (
                'name', 
                'plan_type', 
                'is_active'
            )
        }),
        ('Pricing', {
            'fields': (
                'price_per_month', 
                'price_per_year'
            )
        }),
        ('Limits & Features', {
            'fields': (
                'diagnoses_limit', 
                'features'
            )
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )


# ============================================
# PAYMENT TRANSACTION ADMIN
# ============================================
@admin.register(PaymentTransaction)
class PaymentTransactionAdmin(admin.ModelAdmin):
    list_display = (
        'transaction_id', 
        'user', 
        'plan', 
        'amount', 
        'payment_method',
        'status',
        'created_at'
    )
    list_filter = ('status', 'payment_method', 'created_at')
    search_fields = ('transaction_id', 'user__username', 'user__email')
    readonly_fields = ('created_at', 'completed_at')  # Added readonly fields
    list_per_page = 30
    ordering = ['-created_at']
    
    fieldsets = (
        ('Transaction Information', {
            'fields': (
                'transaction_id', 
                'user', 
                'plan'
            )
        }),
        ('Payment Details', {
            'fields': (
                'amount', 
                'payment_method', 
                'status',
                'receipt_url'
            )
        }),
        ('Timestamps', {
            'fields': ('created_at', 'completed_at'),
            'classes': ('collapse',)
        }),
    )
    
    actions = ['mark_as_completed', 'mark_as_failed', 'mark_as_refunded']
    
    def mark_as_completed(self, request, queryset):
        for transaction in queryset:
            transaction.mark_completed()
    mark_as_completed.short_description = "Mark selected as Completed"
    
    def mark_as_failed(self, request, queryset):
        queryset.update(status='failed')
    mark_as_failed.short_description = "Mark selected as Failed"
    
    def mark_as_refunded(self, request, queryset):
        queryset.update(status='refunded')
    mark_as_refunded.short_description = "Mark selected as Refunded"


# ============================================
# AI MODEL INFO ADMIN
# ============================================
@admin.register(AIModelInfo)
class AIModelInfoAdmin(admin.ModelAdmin):
    list_display = (
        'name', 
        'model_type', 
        'version', 
        'accuracy_percentage',
        'num_classes',
        'is_active'
    )
    list_filter = ('model_type', 'is_active')
    search_fields = ('name', 'model_type', 'description')
    readonly_fields = ('created_at', 'updated_at')  # Added readonly fields
    list_per_page = 20
    
    fieldsets = (
        ('Model Information', {
            'fields': (
                'name', 
                'model_type', 
                'version',
                'description'
            )
        }),
        ('Performance', {
            'fields': (
                'accuracy', 
                'num_classes', 
                'class_names'
            )
        }),
        ('Path & Status', {
            'fields': (
                'model_path', 
                'is_active'
            )
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )
    
    def accuracy_percentage(self, obj):
        return obj.get_accuracy_percentage()
    accuracy_percentage.short_description = "Accuracy"


# ============================================
# ACTIVITY LOG ADMIN
# ============================================
@admin.register(ActivityLog)
class ActivityLogAdmin(admin.ModelAdmin):
    list_display = (
        'user', 
        'action_type', 
        'short_description',
        'ip_address',
        'created_at'
    )
    list_filter = ('action_type', 'created_at')
    search_fields = ('user__username', 'description', 'ip_address')
    readonly_fields = ('created_at',)  # Added readonly fields
    list_per_page = 50
    ordering = ['-created_at']
    
    fieldsets = (
        ('Activity Information', {
            'fields': (
                'user', 
                'action_type', 
                'description'
            )
        }),
        ('Technical Details', {
            'fields': (
                'ip_address', 
                'user_agent',
                'related_record_id',
                'related_model'
            ),
            'classes': ('collapse',)
        }),
        ('Timestamp', {
            'fields': ('created_at',),
        }),
    )
    
    def short_description(self, obj):
        return obj.description[:100] + '...' if len(obj.description) > 100 else obj.description
    short_description.short_description = "Description"
    
    def has_add_permission(self, request):
        return False
    
    def has_delete_permission(self, request, obj=None):
        return request.user.is_superuser


# ============================================
# NOTIFICATION ADMIN
# ============================================
@admin.register(Notification)
class NotificationAdmin(admin.ModelAdmin):
    list_display = (
        'user', 
        'notification_type', 
        'title',
        'is_read',
        'created_at'
    )
    list_filter = ('notification_type', 'is_read', 'created_at')
    search_fields = ('user__username', 'title', 'message')
    readonly_fields = ('created_at',)  # Added readonly fields
    list_per_page = 30
    ordering = ['-created_at']
    
    fieldsets = (
        ('Notification Information', {
            'fields': (
                'user', 
                'notification_type', 
                'title',
                'message'
            )
        }),
        ('Status', {
            'fields': (
                'is_read', 
                'link'
            )
        }),
        ('Timestamp', {
            'fields': ('created_at',),
        }),
    )
    
    actions = ['mark_as_read', 'mark_as_unread']
    
    def mark_as_read(self, request, queryset):
        queryset.update(is_read=True)
    mark_as_read.short_description = "Mark selected as Read"
    
    def mark_as_unread(self, request, queryset):
        queryset.update(is_read=False)
    mark_as_unread.short_description = "Mark selected as Unread"


# ============================================
# USER FEEDBACK ADMIN
# ============================================
@admin.register(UserFeedback)
class UserFeedbackAdmin(admin.ModelAdmin):
    list_display = (
        'user', 
        'diagnosis_record', 
        'rating_stars',
        'is_helpful',
        'created_at'
    )
    list_filter = ('rating', 'is_helpful', 'created_at')
    search_fields = ('user__username', 'comment', 'diagnosis_record__case_id')
    readonly_fields = ('created_at',)  # Added readonly fields
    list_per_page = 30
    ordering = ['-created_at']
    
    fieldsets = (
        ('Feedback Information', {
            'fields': (
                'user', 
                'diagnosis_record',
                'rating',
                'comment'
            )
        }),
        ('Status', {
            'fields': ('is_helpful',)
        }),
        ('Timestamp', {
            'fields': ('created_at',),
        }),
    )
    
    def rating_stars(self, obj):
        return obj.get_rating_stars()
    rating_stars.short_description = "Rating"


# ============================================
# SAVED ANALYSIS ADMIN
# ============================================
@admin.register(SavedAnalysis)
class SavedAnalysisAdmin(admin.ModelAdmin):
    list_display = (
        'user', 
        'diagnosis_record', 
        'created_at'
    )
    list_filter = ('created_at',)
    search_fields = ('user__username', 'diagnosis_record__case_id', 'note')
    readonly_fields = ('created_at',)  # Added readonly fields
    list_per_page = 30
    ordering = ['-created_at']
    
    fieldsets = (
        ('Saved Analysis Information', {
            'fields': (
                'user', 
                'diagnosis_record',
                'note'
            )
        }),
        ('Timestamp', {
            'fields': ('created_at',),
        }),
    )


# ============================================
# SYSTEM SETTING ADMIN
# ============================================
@admin.register(SystemSetting)
class SystemSettingAdmin(admin.ModelAdmin):
    list_display = (
        'key', 
        'setting_type', 
        'value_preview',
        'is_public',
        'updated_at'
    )
    list_filter = ('setting_type', 'is_public')
    search_fields = ('key', 'value', 'description')
    readonly_fields = ('created_at', 'updated_at')  # Added readonly fields
    list_per_page = 20
    
    fieldsets = (
        ('Setting Information', {
            'fields': (
                'key', 
                'value', 
                'setting_type'
            )
        }),
        ('Additional Info', {
            'fields': (
                'description', 
                'is_public'
            )
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )
    
    def value_preview(self, obj):
        if len(obj.value) > 50:
            return obj.value[:50] + '...'
        return obj.value
    value_preview.short_description = "Value"


# ============================================
# USER ADMIN EXTENSION
# ============================================
class UserProfileInline(admin.StackedInline):
    model = UserProfile
    can_delete = False
    verbose_name_plural = 'Profile'
    fieldsets = (
        (None, {
            'fields': (
                'role', 
                'institution', 
                'phone',
                'profile_picture'
            )
        }),
        ('Subscription', {
            'fields': (
                'subscription_plan', 
                'subscription_expiry', 
                'is_subscribed'
            )
        }),
        ('Statistics', {
            'fields': (
                'total_diagnoses', 
                'total_reports'
            )
        }),
    )


# Unregister default User admin and register with profile inline
admin.site.unregister(User)


@admin.register(User)
class CustomUserAdmin(UserAdmin):
    inlines = [UserProfileInline]
    list_display = (
        'username', 
        'email', 
        'first_name', 
        'last_name', 
        'is_staff',
        'date_joined'
    )
    list_filter = ('is_staff', 'is_superuser', 'is_active', 'date_joined')
    search_fields = ('username', 'email', 'first_name', 'last_name')
    ordering = ['-date_joined']
    
    fieldsets = (
        (None, {'fields': ('username', 'password')}),
        ('Personal info', {'fields': ('first_name', 'last_name', 'email')}),
        ('Permissions', {'fields': ('is_active', 'is_staff', 'is_superuser', 'groups', 'user_permissions')}),
        ('Important dates', {'fields': ('last_login', 'date_joined')}),
    )


# ============================================
# ADMIN SITE CONFIGURATION
# ============================================
admin.site.site_header = "X-HViT Medical AI Administration"
admin.site.site_title = "X-HViT Admin Portal"
admin.site.index_title = "Welcome to X-HViT Admin Dashboard"