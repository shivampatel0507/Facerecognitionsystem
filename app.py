import secrets
import os
import uuid
import logging
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_bcrypt import Bcrypt
from functools import wraps
from datetime import date, datetime
import face_recognition
from sqlalchemy import func
import numpy as np

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:FaceRPass@localhost/attendance_db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = secrets.token_hex(32)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Logging setup
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Models
class Standard(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), unique=True, nullable=False)
    users = db.relationship('User', backref='standard', lazy=True)

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password_hash = db.Column(db.LargeBinary, nullable=False)
    role = db.Column(db.String(10), nullable=False)  # 'admin', 'teacher', 'student'
    name = db.Column(db.String(100), nullable=False)
    contact = db.Column(db.String(10), nullable=True)
    email = db.Column(db.String(100), nullable=True)
    standard_id = db.Column(db.Integer, db.ForeignKey('standard.id'), nullable=True)
    roll_number = db.Column(db.Integer, nullable=True)
    uid = db.Column(db.String(18), unique=True, nullable=True)
    face_encoding = db.Column(db.PickleType, nullable=True)
    educational_qualifications = db.Column(db.String(200), nullable=True)

class Attendance(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=db.func.now())
    latitude = db.Column(db.Float, nullable=True)
    longitude = db.Column(db.Float, nullable=True)

# Decorators for role-based access control
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or current_user.role != 'admin':
            flash('You do not have permission to access this page.', 'danger')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def teacher_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or current_user.role != 'teacher':
            flash('You do not have permission to access this page.', 'danger')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def student_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or current_user.role != 'student':
            flash('You do not have permission to access this page.', 'danger')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Helper function to check if a student has marked attendance today
def has_marked_attendance_today(user):
    today = date.today()
    return Attendance.query.filter(
        Attendance.student_id == user.id,
        func.date(Attendance.timestamp) == today
    ).first() is not None

# User loader for Flask-Login
@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

# Login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and bcrypt.check_password_hash(user.password_hash, password):
            login_user(user)
            flash('Login successful!', 'success')
            logger.info(f"User {username} logged in")
            if user.role == 'admin':
                return redirect(url_for('admin_dashboard'))
            elif user.role == 'teacher':
                return redirect(url_for('teacher_dashboard'))
            elif user.role == 'student':
                return redirect(url_for('student_dashboard'))
        else:
            flash('Invalid username or password.', 'danger')
            logger.warning(f"Failed login attempt for username: {username}")
    return render_template('login.html')

# Logout route
@app.route('/logout')
@login_required
def logout():
    username = current_user.username
    logout_user()
    flash('You have been logged out.', 'info')
    logger.info(f"User {username} logged out")
    return redirect(url_for('login'))

# Admin dashboard
@app.route('/admin/dashboard')
@admin_required
def admin_dashboard():
    teachers = User.query.filter_by(role='teacher').all()
    return render_template('admin/dashboard.html', teachers=teachers)

# Add teacher
@app.route('/admin/add_teacher', methods=['GET', 'POST'])
@admin_required
def add_teacher():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        name = request.form['name']
        contact = request.form['contact']
        email = request.form['email']
        standard_id = request.form['standard_id']
        educational_qualifications = request.form['educational_qualifications']
        # Validations
        if not all([username, password, name, contact, email, standard_id, educational_qualifications]):
            flash('All fields are required.', 'danger')
            return redirect(request.url)
        if not name.isalpha():
            flash('Name must contain only letters.', 'danger')
            return redirect(request.url)
        if not (contact.isdigit() and len(contact) == 10):
            flash('Contact must be exactly 10 digits.', 'danger')
            return redirect(request.url)
        if '@gmail.com' not in email:
            flash('Email must be a valid Gmail address.', 'danger')
            return redirect(request.url)
        if User.query.filter_by(username=username).first():
            flash('Username already exists.', 'danger')
            return redirect(request.url)
        try:
            standard_id_int = int(standard_id)
        except ValueError:
            flash('Invalid standard.', 'danger')
            return redirect(request.url)
        hashed_password = bcrypt.generate_password_hash(password).encode('utf-8')
        teacher = User(username=username, password_hash=hashed_password, role='teacher',
                       name=name, contact=contact, email=email,
                       standard_id=standard_id_int, educational_qualifications=educational_qualifications)
        db.session.add(teacher)
        db.session.commit()
        flash('Teacher added successfully.', 'success')
        logger.info(f"Teacher {username} added by Admin")
        return redirect(url_for('admin_dashboard'))
    standards = Standard.query.all()
    if not standards:
        db.session.add(Standard(name='1st Std'))
        db.session.add(Standard(name='2nd Std'))
        db.session.commit()
        standards = Standard.query.all()
    return render_template('admin/add_teacher.html', standards=standards)

# Edit teacher
@app.route('/admin/edit_teacher/<int:teacher_id>', methods=['GET', 'POST'])
@admin_required
def edit_teacher(teacher_id):
    teacher = User.query.get_or_404(teacher_id)
    if teacher.role != 'teacher':
        flash('Invalid teacher.', 'danger')
        return redirect(url_for('admin_dashboard'))
    if request.method == 'POST':
        teacher.username = request.form['username']
        if request.form['password']:
            teacher.password_hash = bcrypt.generate_password_hash(request.form['password']).encode('utf-8')
        teacher.name = request.form['name']
        teacher.contact = request.form['contact']
        teacher.email = request.form['email']
        try:
            teacher.standard_id = int(request.form['standard_id'])
        except ValueError:
            flash('Invalid standard.', 'danger')
            return redirect(request.url)
        teacher.educational_qualifications = request.form['educational_qualifications']
        db.session.commit()
        flash('Teacher updated successfully.', 'success')
        logger.info(f"Teacher {teacher.username} updated by Admin")
        return redirect(url_for('admin_dashboard'))
    standards = Standard.query.all()
    return render_template('admin/edit_teacher.html', teacher=teacher, standards=standards)

# Delete teacher
@app.route('/admin/delete_teacher/<int:teacher_id>', methods=['POST'])
@admin_required
def delete_teacher(teacher_id):
    teacher = User.query.get_or_404(teacher_id)
    if teacher.role != 'teacher':
        flash('Invalid teacher.', 'danger')
    else:
        db.session.delete(teacher)
        db.session.commit()
        flash('Teacher deleted successfully.', 'success')
        logger.info(f"Teacher {teacher.username} deleted by Admin")
    return redirect(url_for('admin_dashboard'))

# Teacher dashboard
@app.route('/teacher/dashboard')
@teacher_required
def teacher_dashboard():
    students = User.query.filter_by(role='student', standard_id=current_user.standard_id).all()
    return render_template('teacher/dashboard.html', students=students)

# Add student
@app.route('/teacher/add_student', methods=['GET', 'POST'])
@teacher_required
def add_student():
    if request.method == 'POST':
        roll_number = request.form['roll_number']
        name = request.form['name']
        contact = request.form['contact']
        uid = request.form['uid']
        password = request.form['password']
        if not all([roll_number, name, contact, uid, password]):
            flash('All fields are required.', 'danger')
            return redirect(request.url)
        if not name.isalpha():
            flash('Name must contain only letters.', 'danger')
            return redirect(request.url)
        if not (contact.isdigit() and len(contact) == 10):
            flash('Contact must be exactly 10 digits.', 'danger')
            return redirect(request.url)
        if not (uid.isdigit() and len(uid) == 18):
            flash('UID must be exactly 18 digits.', 'danger')
            return redirect(request.url)
        if User.query.filter_by(uid=uid).first():
            flash('UID already exists.', 'danger')
            return redirect(request.url)
        hashed_password = bcrypt.generate_password_hash(password).encode('utf-8')
        student = User(username=uid, role='student', name=name, contact=contact,
                       standard_id=current_user.standard_id, roll_number=roll_number,
                       uid=uid, password_hash=hashed_password)
        db.session.add(student)
        db.session.commit()
        flash('Student added successfully. Please capture face.', 'success')
        logger.info(f"Student {uid} added by Teacher {current_user.username}")
        return redirect(url_for('capture_face', student_id=student.id))
    return render_template('teacher/add_student.html')

# Edit student
@app.route('/teacher/edit_student/<int:student_id>', methods=['GET', 'POST'])
@teacher_required
def edit_student(student_id):
    student = User.query.get_or_404(student_id)
    if student.role != 'student' or student.standard_id != current_user.standard_id:
        flash('Invalid student.', 'danger')
        return redirect(url_for('teacher_dashboard'))
    if request.method == 'POST':
        student.roll_number = request.form['roll_number']
        student.name = request.form['name']
        student.contact = request.form['contact']
        if request.form['password']:
            student.password_hash = bcrypt.generate_password_hash(request.form['password']).encode('utf-8')
        db.session.commit()
        flash('Student updated successfully.', 'success')
        logger.info(f"Student {student.uid} updated by Teacher {current_user.username}")
        return redirect(url_for('teacher_dashboard'))
    return render_template('teacher/edit_student.html', student=student)

# Delete student
@app.route('/teacher/delete_student/<int:student_id>', methods=['POST'])
@teacher_required
def delete_student(student_id):
    student = User.query.get_or_404(student_id)
    if student.role != 'student' or student.standard_id != current_user.standard_id:
        flash('Invalid student.', 'danger')
    else:
        db.session.delete(student)
        db.session.commit()
        flash('Student deleted successfully.', 'success')
        logger.info(f"Student {student.uid} deleted by Teacher {current_user.username}")
    return redirect(url_for('teacher_dashboard'))

# Capture face for student
@app.route('/teacher/capture_face/<int:student_id>', methods=['GET', 'POST'])
@teacher_required
def capture_face(student_id):
    student = User.query.get_or_404(student_id)
    if student.role != 'student' or student.standard_id != current_user.standard_id:
        flash('Invalid student.', 'danger')
        return redirect(url_for('teacher_dashboard'))
    if request.method == 'POST':
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        file = request.files['image']
        # Use a unique temporary filename
        temp_filename = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_{uuid.uuid4().hex}.jpg")
        file.save(temp_filename)
        try:
            image = face_recognition.load_image_file(temp_filename)
            face_encodings = face_recognition.face_encodings(image)
            if not face_encodings:
                return jsonify({'error': 'No face detected'}), 400
            student.face_encoding = face_encodings[0]
            db.session.commit()
            return jsonify({'success': True}), 200
        except Exception as e:
            logger.error(f"Error capturing face for student {student.uid}: {e}")
            return jsonify({'error': 'An error occurred. Please try again.'}), 500
        finally:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
    return render_template('teacher/capture_face.html', student=student)

# Get student attendance for modal
@app.route('/teacher/get_student_attendance/<int:student_id>', methods=['GET'])
@teacher_required
def get_student_attendance(student_id):
    student = User.query.get_or_404(student_id)
    if student.role != 'student' or student.standard_id != current_user.standard_id:
        return jsonify({'error': 'Invalid student'}), 403
    attendance_records = Attendance.query.filter_by(student_id=student_id).order_by(Attendance.timestamp.desc()).all()
    attendance_data = [{
        'date': record.timestamp.strftime('%Y-%m-%d'),
        'time': record.timestamp.strftime('%H:%M:%S')
    } for record in attendance_records]
    return jsonify({'attendance': attendance_data})

# Student dashboard
@app.route('/student/dashboard')
@student_required
def student_dashboard():
    marked_today = has_marked_attendance_today(current_user)
    attendance_records = Attendance.query.filter_by(student_id=current_user.id).order_by(Attendance.timestamp.desc()).all()
    return render_template('student/dashboard.html', marked_today=marked_today, attendance_records=attendance_records)

# Mark attendance with blink detection
@app.route('/student/mark_attendance', methods=['GET', 'POST'])
@student_required
def mark_attendance():
    if request.method == 'POST':
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        file = request.files['image']
        # Use a unique temporary filename
        temp_filename = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_{uuid.uuid4().hex}.jpg")
        file.save(temp_filename)
        try:
            image = face_recognition.load_image_file(temp_filename)
            face_encodings = face_recognition.face_encodings(image)
            if not face_encodings:
                return jsonify({'error': 'No face detected'}), 400
            detected_encoding = face_encodings[0]
            if current_user.face_encoding is None:
                return jsonify({'error': 'No face encoding registered'}), 400
            match = face_recognition.compare_faces([current_user.face_encoding], detected_encoding)[0]
            if not match:
                return jsonify({'error': 'Face not recognized'}), 400
            if has_marked_attendance_today(current_user):
                return jsonify({'error': 'Attendance already marked today'}), 400
            latitude = request.form.get('latitude')
            longitude = request.form.get('longitude')
            try:
                latitude = float(latitude) if latitude else None
                longitude = float(longitude) if longitude else None
            except ValueError:
                logger.warning(f"Invalid geolocation data for student {current_user.uid}: lat={latitude}, lon={longitude}")
                latitude, longitude = None, None
            attendance = Attendance(
                student_id=current_user.id,
                latitude=latitude,
                longitude=longitude
            )
            db.session.add(attendance)
            db.session.commit()
            logger.info(f"Attendance marked for student {current_user.uid}")
            return jsonify({'success': True}), 200
        except Exception as e:
            logger.error(f"Error marking attendance for student {current_user.uid}: {e}")
            return jsonify({'error': str(e)}), 500
        finally:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
    else:
        if has_marked_attendance_today(current_user):
            flash('You have already marked attendance today.', 'info')
            return redirect(url_for('student_dashboard'))
        return render_template('student/attendance.html')

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        # Create admin if not exists
        admin = User.query.filter_by(role='admin').first()
        if not admin:
            hashed_password = bcrypt.generate_password_hash('adm4425').encode('utf-8')
            admin = User(username='Admin', role='admin', name='Administrator', password_hash=hashed_password)
            db.session.add(admin)
            db.session.commit()
            logger.info("Admin account created")
    app.run(debug=True)