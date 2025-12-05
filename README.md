

### Python 3.11 (recomendado)
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate
python -m pip install --upgrade pip setuptools wheel
pip install pandas sqlalchemy joblib scikit-learn
```

### Entrenamiento
```powershell
python ml/train_random_forest_citas.py
```


Ejecutar con este comando

python .\ml\predict_slots.py --db_url "postgresql://postgres:@localhost:5432/consultorio" --prioridades "muy_urgente,alta,media" --weeks 1 --edad 30 --dias_desde_ultima 14 --output_file "citas_sugeridas.csv" --file_type "csv"


BASE DE DATOS

CREATE TABLE auth_group (
    id SERIAL PRIMARY KEY,
    name VARCHAR NOT NULL
);

CREATE TABLE auth_group_permissions (
    id BIGINT PRIMARY KEY,
    group_id INTEGER NOT NULL REFERENCES auth_group(id),
    permission_id INTEGER NOT NULL
);

CREATE TABLE auth_permission (
    id SERIAL PRIMARY KEY,
    name VARCHAR NOT NULL,
    content_type_id INTEGER NOT NULL,
    codename VARCHAR NOT NULL
);

CREATE TABLE auth_user (
    id SERIAL PRIMARY KEY,
    password VARCHAR NOT NULL,
    last_login TIMESTAMPTZ,
    is_superuser BOOLEAN NOT NULL,
    username VARCHAR NOT NULL,
    first_name VARCHAR NOT NULL,
    last_name VARCHAR NOT NULL,
    email VARCHAR NOT NULL,
    is_staff BOOLEAN NOT NULL,
    is_active BOOLEAN NOT NULL,
    date_joined TIMESTAMPTZ NOT NULL
);

CREATE TABLE auth_user_groups (
    id BIGINT PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES auth_user(id),
    group_id INTEGER NOT NULL REFERENCES auth_group(id)
);

CREATE TABLE auth_user_user_permissions (
    id BIGINT PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES auth_user(id),
    permission_id INTEGER NOT NULL
);

CREATE TABLE citas (
    id_cita SERIAL PRIMARY KEY,
    fecha_hora TIMESTAMPTZ NOT NULL,
    estado VARCHAR NOT NULL,
    motivo TEXT,
    observaciones TEXT,
    creada_en TIMESTAMPTZ NOT NULL,
    actualizada_en TIMESTAMPTZ NOT NULL,
    id_paciente INTEGER NOT NULL,
    id_psicologo INTEGER NOT NULL
);

CREATE TABLE citas_sugeridas (
    id_sugerencia SERIAL PRIMARY KEY,
    fecha_hora_sugerida TIMESTAMPTZ,
    score NUMERIC,
    orden INTEGER NOT NULL,
    origen VARCHAR NOT NULL,
    estado VARCHAR NOT NULL,
    features JSONB,
    created_at TIMESTAMPTZ NOT NULL,
    id_cita_aceptada INTEGER,
    id_modelo INTEGER,
    id_paciente INTEGER,
    id_psicologo INTEGER,
    fecha TIMESTAMP,
    hora INTEGER,
    prioridad VARCHAR,
    dia VARCHAR,
    fecha_str VARCHAR,
    hora_str VARCHAR
);

CREATE TABLE control_acceso (
    id_evento SERIAL PRIMARY KEY,
    fecha_hora TIMESTAMPTZ NOT NULL,
    modulo VARCHAR NOT NULL,
    accion VARCHAR NOT NULL,
    resultado BOOLEAN NOT NULL,
    id_usuario INTEGER NOT NULL
);

CREATE TABLE diario_emocional (
    id_diario SERIAL PRIMARY KEY,
    descripcion TEXT NOT NULL,
    titulo_dia VARCHAR,
    fecha DATE NOT NULL,
    id_paciente INTEGER NOT NULL,
    id_psicologo INTEGER
);

CREATE TABLE diario_permisos (
    id_permiso SERIAL PRIMARY KEY,
    permitido BOOLEAN NOT NULL,
    actualizado_en TIMESTAMPTZ NOT NULL,
    id_paciente INTEGER NOT NULL,
    id_psicologo INTEGER NOT NULL
);

CREATE TABLE disponibilidad (
    id_paciente INTEGER REFERENCES pacientes(id_paciente),
    nombre VARCHAR,
    apellido VARCHAR,
    dia_semana VARCHAR,
    hora_inicio TIME WITHOUT TIME ZONE,
    hora_fin TIME WITHOUT TIME ZONE,
    id_psicologo INTEGER,
    prioridad VARCHAR,
    fecha DATE
);

CREATE TABLE disponibilidades (
    id_disponibilidad SERIAL PRIMARY KEY,
    dia_semana SMALLINT NOT NULL,
    hora_inicio TIME WITHOUT TIME ZONE NOT NULL,
    hora_fin TIME WITHOUT TIME ZONE NOT NULL,
    activo BOOLEAN NOT NULL,
    creado_en TIMESTAMPTZ NOT NULL,
    id_paciente INTEGER NOT NULL
);

CREATE TABLE django_admin_log (
    id SERIAL PRIMARY KEY,
    action_time TIMESTAMPTZ NOT NULL,
    object_id TEXT,
    object_repr VARCHAR NOT NULL,
    action_flag SMALLINT NOT NULL,
    change_message TEXT NOT NULL,
    content_type_id INTEGER,
    user_id INTEGER NOT NULL
);

CREATE TABLE django_content_type (
    id SERIAL PRIMARY KEY,
    app_label VARCHAR NOT NULL,
    model VARCHAR NOT NULL
);

CREATE TABLE django_migrations (
    id BIGINT PRIMARY KEY,
    app VARCHAR NOT NULL,
    name VARCHAR NOT NULL,
    applied TIMESTAMPTZ NOT NULL
);

CREATE TABLE django_session (
    session_key VARCHAR PRIMARY KEY,
    session_data TEXT NOT NULL,
    expire_date TIMESTAMPTZ NOT NULL
);

CREATE TABLE entrenamientos_ml (
    id_entrenamiento SERIAL PRIMARY KEY,
    fecha_inicio TIMESTAMPTZ,
    fecha_fin TIMESTAMPTZ,
    dataset VARCHAR NOT NULL,
    hiperparametros JSONB,
    metricas JSONB,
    id_modelo INTEGER NOT NULL
);

CREATE TABLE modelos_ml (
    id_modelo SERIAL PRIMARY KEY,
    tipo VARCHAR NOT NULL,
    nombre VARCHAR NOT NULL,
    version VARCHAR NOT NULL,
    fecha_entrenamiento TIMESTAMPTZ,
    descripcion TEXT,
    tamano_bytes BIGINT,
    formato VARCHAR
);

CREATE TABLE pacientes (
    id_paciente SERIAL PRIMARY KEY,
    nombre VARCHAR NOT NULL,
    apellido VARCHAR NOT NULL,
    direccion TEXT,
    fecha_nacimiento DATE,
    edad SMALLINT,
    trastorno_salud_mental TEXT,
    fecha_registro DATE NOT NULL,
    prioridad_clinica VARCHAR NOT NULL,
    id_psicologo INTEGER,
    id_usuario INTEGER,
    email VARCHAR,
    numero_celular VARCHAR,
    prioridad VARCHAR
);

CREATE TABLE pln_procesos (
    id_proceso SERIAL PRIMARY KEY,
    fecha_proceso TIMESTAMPTZ NOT NULL,
    resumen_generado TEXT,
    calidad NUMERIC,
    id_modelo INTEGER,
    id_sesion INTEGER NOT NULL
);

CREATE TABLE psicologos (
    id_psicologo SERIAL PRIMARY KEY,
    nombre VARCHAR NOT NULL,
    apellido VARCHAR NOT NULL,
    fecha_nacimiento DATE,
    edad SMALLINT,
    numero_matricula VARCHAR NOT NULL,
    direccion TEXT,
    numero_contacto VARCHAR,
    correo_electronico VARCHAR,
    lugar_expedicion_certificado TEXT
);

CREATE TABLE roles (
    id_rol SERIAL PRIMARY KEY,
    nombre VARCHAR NOT NULL,
    descripcion VARCHAR
);

CREATE TABLE sesiones (
    id_sesion SERIAL PRIMARY KEY,
    fecha DATE,
    grabacion_audio VARCHAR,
    transcripcion TEXT,
    resumen TEXT,
    id_cita INTEGER
);

CREATE TABLE solicitudes_cita (
    id_solicitud SERIAL PRIMARY KEY,
    tipo VARCHAR NOT NULL,
    motivo TEXT,
    estado VARCHAR NOT NULL,
    creado_en TIMESTAMPTZ NOT NULL,
    id_cita INTEGER NOT NULL,
    id_paciente INTEGER NOT NULL
);

CREATE TABLE sugerencia_cita (
    id BIGINT PRIMARY KEY,
    inicio TIMESTAMPTZ NOT NULL,
    fin TIMESTAMPTZ NOT NULL,
    score NUMERIC,
    fuente VARCHAR NOT NULL,
    estado VARCHAR NOT NULL,
    notas TEXT,
    creado_en TIMESTAMPTZ,
    actualizado_en TIMESTAMPTZ,
    paciente_id INTEGER,
    psicologo_id INTEGER NOT NULL
);

CREATE TABLE usuarios (
    id_usuario SERIAL PRIMARY KEY,
    correo_electronico VARCHAR NOT NULL,
    contrasena VARCHAR NOT NULL,
    id_rol INTEGER NOT NULL
);







INSERTS PARA HACER EL RF
-- Insertar paciente 1
INSERT INTO pacientes (id_paciente, nombre, apellido, direccion, fecha_nacimiento, edad, trastorno_salud_mental, fecha_registro, prioridad_clinica, id_psicologo, id_usuario, email, numero_celular)
VALUES 
(101, 'Samuel', 'Alvarezzz', '11111111111111111', '1990-05-01', 30, 'Ansiedad', '2025-12-01', 'urgente', 1, 1, 'samuel@email.com', '9876543210');

-- Insertar paciente 2
INSERT INTO pacientes (id_paciente, nombre, apellido, direccion, fecha_nacimiento, edad, trastorno_salud_mental, fecha_registro, prioridad_clinica, id_psicologo, id_usuario, email, numero_celular)
VALUES 
(102, 'juan', 'juan', 'hhh', '1992-07-21', 28, 'Depresión', '2025-12-01', 'alta', 1, 1, 'juan@email.com', '9876543220');

-- Insertar paciente 3
INSERT INTO pacientes (id_paciente, nombre, apellido, direccion, fecha_nacimiento, edad, trastorno_salud_mental, fecha_registro, prioridad_clinica, id_psicologo, id_usuario, email, numero_celular)
VALUES 
(103, 'Samuel Franz', 'Alvarez Morillas', 'Llojeta Rosal Calle rosa verde', '1985-03-14', 40, 'Estrés Post Traumático', '2025-12-01', 'media', 1, 1, 'samuelf@email.com', '9876543230');

-- Insertar paciente 4
INSERT INTO pacientes (id_paciente, nombre, apellido, direccion, fecha_nacimiento, edad, trastorno_salud_mental, fecha_registro, prioridad_clinica, id_psicologo, id_usuario, email, numero_celular)
VALUES 
(104, 'Juan', 'Perez', 'Llojeta 2', '1980-11-23', 45, 'Bipolaridad', '2025-12-01', 'regula', 1, 1, 'juanp@email.com', '9876543240');

-- Insertar paciente 5
INSERT INTO pacientes (id_paciente, nombre, apellido, direccion, fecha_nacimiento, edad, trastorno_salud_mental, fecha_registro, prioridad_clinica, id_psicologo, id_usuario, email, numero_celular)
VALUES 
(105, 'Samuel Franz', 'Alvarez Morillas', 'Llojeta Rosal Calle rosa verde', '1983-09-30', 38, 'Trastorno Obsesivo Compulsivo', '2025-12-01', 'baja', 1, 1, 'samuelb@email.com', '9876543250');

-- Insertar paciente 6
INSERT INTO pacientes (id_paciente, nombre, apellido, direccion, fecha_nacimiento, edad, trastorno_salud_mental, fecha_registro, prioridad_clinica, id_psicologo, id_usuario, email, numero_celular)
VALUES 
(106, 'juancito', 'perez', 'calle 1', '1995-06-15', 25, 'Ansiedad generalizada', '2025-12-01', 'baja', 1, 1, 'juancito@email.com', '9876543260');
