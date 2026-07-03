#include "audio_device_utils.h"
#include "realtime_session.h"

#include <QApplication>
#include <QCheckBox>
#include <QComboBox>
#include <QDateTime>
#include <QDesktopServices>
#include <QFileDialog>
#include <QFormLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QMainWindow>
#include <QMetaObject>
#include <QPushButton>
#include <QStatusBar>
#include <QTextEdit>
#include <QUrl>
#include <QVariant>
#include <QVBoxLayout>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#ifndef VOX_PROJECT_ROOT
#define VOX_PROJECT_ROOT "."
#endif

namespace {

QString to_qstring(const std::string & value) {
    return QString::fromUtf8(value.data(), static_cast<qsizetype>(value.size()));
}

int32_t current_device_id(const QComboBox * combo) {
    const QVariant data = combo->currentData();
    return data.isValid() ? static_cast<int32_t>(data.toInt()) : -1;
}

class MainWindow final : public QMainWindow {
public:
    MainWindow() {
        setWindowTitle("Vox.cpp");
        resize(980, 720);

        auto * central = new QWidget(this);
        auto * root = new QVBoxLayout(central);
        root->setContentsMargins(16, 16, 16, 16);
        root->setSpacing(12);

        auto * device_group = new QGroupBox("Audio Devices", central);
        auto * device_layout = new QVBoxLayout(device_group);

        auto * capture_row = new QHBoxLayout();
        capture_row->addWidget(new QLabel("Microphone", device_group));
        capture_devices_ = new QComboBox(device_group);
        capture_row->addWidget(capture_devices_, 1);
        auto * refresh_button = new QPushButton("Refresh", device_group);
        capture_row->addWidget(refresh_button);
        device_layout->addLayout(capture_row);

        auto * output_row = new QHBoxLayout();
        output_row->addWidget(new QLabel("Virtual mic output", device_group));
        output_devices_ = new QComboBox(device_group);
        output_row->addWidget(output_devices_, 1);
        auto * cable_button = new QPushButton("VB-CABLE", device_group);
        output_row->addWidget(cable_button);
        device_layout->addLayout(output_row);

        cable_notice_ = new QLabel(device_group);
        cable_notice_->setWordWrap(true);
        device_layout->addWidget(cable_notice_);
        root->addWidget(device_group);

        auto * model_group = new QGroupBox("Models", central);
        auto * model_layout = new QFormLayout(model_group);
        asr_model_ = add_path_row(model_layout, model_group, "ASR GGUF",
                                  "models/asr/qwen3-asr-1.7b/Qwen3-ASR-1.7B-Q8_0.gguf");
        asr_mmproj_ = add_path_row(model_layout, model_group, "ASR mmproj",
                                   "models/asr/qwen3-asr-1.7b/mmproj-Qwen3-ASR-1.7B-Q8_0.gguf");
        translation_model_ = add_path_row(model_layout, model_group, "Translation GGUF",
                                          "models/translate/HY-MT1.5-1.8B-Q4_K_M.gguf");
        tts_model_ = add_path_row(model_layout, model_group, "TTS GGUF",
                                  "models/tts/cosyvoice3/cosyvoice3-llm-q4_k.gguf");
        root->addWidget(model_group);

        auto * settings_group = new QGroupBox("Runtime", central);
        auto * settings_layout = new QFormLayout(settings_group);

        source_language_ = new QLineEdit("auto", settings_group);
        target_language_ = new QLineEdit("Chinese", settings_group);
        tts_voice_ = new QLineEdit("zero_shot", settings_group);
        gpu_enabled_ = new QCheckBox("GPU acceleration", settings_group);
        gpu_enabled_->setChecked(true);
        speak_partials_ = new QCheckBox("Speak partial transcripts", settings_group);
        speak_partials_->setChecked(false);

        settings_layout->addRow("Source language", source_language_);
        settings_layout->addRow("Target language", target_language_);
        settings_layout->addRow("CosyVoice3 voice", tts_voice_);
        settings_layout->addRow("", gpu_enabled_);
        settings_layout->addRow("", speak_partials_);
        root->addWidget(settings_group);

        auto * control_row = new QHBoxLayout();
        start_button_ = new QPushButton("Start", central);
        stop_button_ = new QPushButton("Stop", central);
        stop_button_->setEnabled(false);
        control_row->addWidget(start_button_);
        control_row->addWidget(stop_button_);
        control_row->addStretch(1);
        rms_label_ = new QLabel("RMS 0.0000  Peak 0.0000  Queue 0 KB", central);
        control_row->addWidget(rms_label_);
        root->addLayout(control_row);

        log_ = new QTextEdit(central);
        log_->setReadOnly(true);
        log_->setLineWrapMode(QTextEdit::NoWrap);
        root->addWidget(log_, 1);

        setCentralWidget(central);
        statusBar()->showMessage("Stopped");

        connect(refresh_button, &QPushButton::clicked, this, [this] {
            refresh_devices();
        });
        connect(cable_button, &QPushButton::clicked, this, [] {
            QDesktopServices::openUrl(QUrl("https://vb-audio.com/Cable/"));
        });
        connect(start_button_, &QPushButton::clicked, this, [this] {
            start_session();
        });
        connect(stop_button_, &QPushButton::clicked, this, [this] {
            append_log(QStringLiteral("Stopping..."));
            session_.request_stop();
            stop_button_->setEnabled(false);
        });

        configure_session_callbacks();
        refresh_devices();
    }

    ~MainWindow() override {
        session_.stop();
    }

private:
    QLineEdit * add_path_row(QFormLayout * form, QWidget * parent, const QString & label, const QString & value) {
        auto * row = new QWidget(parent);
        auto * layout = new QHBoxLayout(row);
        layout->setContentsMargins(0, 0, 0, 0);

        auto * edit = new QLineEdit(value, row);
        auto * browse = new QPushButton("Browse", row);
        layout->addWidget(edit, 1);
        layout->addWidget(browse);

        connect(browse, &QPushButton::clicked, this, [this, edit] {
            const QString path = QFileDialog::getOpenFileName(
                this,
                "Select model",
                QString::fromUtf8(VOX_PROJECT_ROOT),
                "GGUF and models (*.gguf *.bin);;All files (*)");
            if (!path.isEmpty()) {
                edit->setText(path);
            }
        });

        form->addRow(label, row);
        return edit;
    }

    void configure_session_callbacks() {
        session_.on_status = [this](vox::app::RealtimeStatusEvent event) {
            QMetaObject::invokeMethod(this, [this, event = std::move(event)] {
                set_running_ui(event.running);
                statusBar()->showMessage(to_qstring(event.message));
                append_log(to_qstring(event.message));
            }, Qt::QueuedConnection);
        };
        session_.on_text = [this](vox::app::RealtimeTextEvent event) {
            QMetaObject::invokeMethod(this, [this, event = std::move(event)] {
                append_log(QString("[%1] %2: %3 (%4 ms)")
                               .arg(static_cast<qulonglong>(event.chunk_index))
                               .arg(to_qstring(event.stage))
                               .arg(to_qstring(event.text))
                               .arg(static_cast<qlonglong>(event.elapsed_ms)));
            }, Qt::QueuedConnection);
        };
        session_.on_audio = [this](vox::app::RealtimeAudioEvent event) {
            QMetaObject::invokeMethod(this, [this, event] {
                rms_label_->setText(QString("RMS %1  Peak %2  Queue %3 KB")
                                        .arg(event.rms, 0, 'f', 4)
                                        .arg(event.peak, 0, 'f', 4)
                                        .arg(event.queued_output_bytes / 1024));
            }, Qt::QueuedConnection);
        };
        session_.on_error = [this](std::string error) {
            QMetaObject::invokeMethod(this, [this, error = std::move(error)] {
                append_log(QStringLiteral("ERROR: ") + to_qstring(error));
            }, Qt::QueuedConnection);
        };
    }

    void refresh_devices() {
        const int32_t previous_capture = current_device_id(capture_devices_);
        const int32_t previous_output = current_device_id(output_devices_);
        const bool had_previous_output = output_devices_->count() > 0;

        capture_devices_->clear();
        for (const vox::app::AudioDeviceInfo & device : vox::app::list_audio_devices(true)) {
            capture_devices_->addItem(to_qstring(device.name), device.id);
        }
        select_device(capture_devices_, previous_capture);

        bool found_virtual_cable = false;
        int virtual_cable_index = -1;
        output_devices_->clear();
        const std::vector<vox::app::AudioDeviceInfo> outputs = vox::app::list_audio_devices(false);
        for (const vox::app::AudioDeviceInfo & device : outputs) {
            output_devices_->addItem(to_qstring(device.name), device.id);
            if (device.likely_virtual_cable && virtual_cable_index < 0) {
                virtual_cable_index = output_devices_->count() - 1;
                found_virtual_cable = true;
            }
        }
        if ((!had_previous_output || !select_device(output_devices_, previous_output)) && virtual_cable_index >= 0) {
            output_devices_->setCurrentIndex(virtual_cable_index);
        }

        if (found_virtual_cable) {
            cable_notice_->setText("Detected a likely virtual cable playback endpoint. Select CABLE Input here, then select CABLE Output as the microphone in Steam or the target game.");
            cable_notice_->setStyleSheet("color: #2f6f3e;");
        } else {
            cable_notice_->setText("VB-CABLE was not detected. Install VB-CABLE, reboot if the installer asks, then refresh devices. Vox writes audio to CABLE Input; games read from CABLE Output.");
            cable_notice_->setStyleSheet("color: #8a5a00;");
        }
    }

    bool select_device(QComboBox * combo, int32_t id) {
        for (int i = 0; i < combo->count(); ++i) {
            if (combo->itemData(i).toInt() == id) {
                combo->setCurrentIndex(i);
                return true;
            }
        }
        if (combo->count() > 0) {
            combo->setCurrentIndex(0);
        }
        return false;
    }

    void start_session() {
        if (session_.running()) {
            return;
        }

        vox::app::RealtimeSessionConfig config;
        config.project_root = QString::fromUtf8(VOX_PROJECT_ROOT).toStdString();
        config.asr_engine = vox::app::RealtimeAsrEngine::Qwen3;
        config.tts_engine = vox::app::RealtimeTtsEngine::CosyVoice3;
        config.asr_model_path = asr_model_->text().toStdString();
        config.asr_mmproj_path = asr_mmproj_->text().toStdString();
        config.translation_model_path = translation_model_->text().toStdString();
        config.tts_model_path = tts_model_->text().toStdString();
        config.language = source_language_->text().toStdString();
        config.target_language = target_language_->text().toStdString();
        config.tts_language = config.target_language;
        config.tts_voice = tts_voice_->text().toStdString();
        config.capture_device_id = current_device_id(capture_devices_);
        config.playback_device_id = current_device_id(output_devices_);
        config.use_gpu = gpu_enabled_->isChecked();
        config.speak_partials = speak_partials_->isChecked();

        append_log(QStringLiteral("Starting..."));
        set_running_ui(true);
        if (!session_.start(std::move(config))) {
            append_log(QStringLiteral("Session is already running."));
            set_running_ui(false);
        }
    }

    void set_running_ui(bool running) {
        start_button_->setEnabled(!running);
        stop_button_->setEnabled(running);
    }

    void append_log(const QString & message) {
        const QString time = QDateTime::currentDateTime().toString("HH:mm:ss");
        log_->append(QStringLiteral("[%1] %2").arg(time).arg(message));
    }

    QComboBox * capture_devices_ = nullptr;
    QComboBox * output_devices_ = nullptr;
    QLabel * cable_notice_ = nullptr;
    QLabel * rms_label_ = nullptr;
    QLineEdit * asr_model_ = nullptr;
    QLineEdit * asr_mmproj_ = nullptr;
    QLineEdit * translation_model_ = nullptr;
    QLineEdit * tts_model_ = nullptr;
    QLineEdit * source_language_ = nullptr;
    QLineEdit * target_language_ = nullptr;
    QLineEdit * tts_voice_ = nullptr;
    QCheckBox * gpu_enabled_ = nullptr;
    QCheckBox * speak_partials_ = nullptr;
    QPushButton * start_button_ = nullptr;
    QPushButton * stop_button_ = nullptr;
    QTextEdit * log_ = nullptr;
    vox::app::RealtimeSession session_;
};

} // namespace

int main(int argc, char ** argv) {
    QApplication app(argc, argv);
    MainWindow window;
    window.show();
    return app.exec();
}
