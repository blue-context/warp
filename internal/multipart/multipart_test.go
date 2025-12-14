package multipart

import (
	"bytes"
	"io"
	"mime/multipart"
	"strings"
	"testing"
)

func TestCreateFormFile(t *testing.T) {
	tests := []struct {
		name      string
		fieldName string
		filename  string
		fileData  string
		fields    map[string]string
		wantErr   bool
	}{
		{
			name:      "valid multipart form",
			fieldName: "file",
			filename:  "test.txt",
			fileData:  "Hello, World!",
			fields: map[string]string{
				"model":    "test-model",
				"language": "en",
			},
			wantErr: false,
		},
		{
			name:      "empty field name",
			fieldName: "",
			filename:  "test.txt",
			fileData:  "data",
			fields:    map[string]string{},
			wantErr:   true,
		},
		{
			name:      "empty filename",
			fieldName: "file",
			filename:  "",
			fileData:  "data",
			fields:    map[string]string{},
			wantErr:   true,
		},
		{
			name:      "nil file reader",
			fieldName: "file",
			filename:  "test.txt",
			fileData:  "",
			fields:    map[string]string{},
			wantErr:   true,
		},
		{
			name:      "empty fields map",
			fieldName: "file",
			filename:  "test.txt",
			fileData:  "test data",
			fields:    map[string]string{},
			wantErr:   false,
		},
		{
			name:      "special characters in filename",
			fieldName: "file",
			filename:  "test file (1).mp3",
			fileData:  "audio data",
			fields: map[string]string{
				"model": "whisper-1",
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var fileReader io.Reader
			if tt.fileData != "" || !tt.wantErr {
				fileReader = strings.NewReader(tt.fileData)
			}

			body, contentType, err := CreateFormFile(tt.fieldName, tt.filename, fileReader, tt.fields)

			if (err != nil) != tt.wantErr {
				t.Errorf("CreateFormFile() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if tt.wantErr {
				return
			}

			// Verify content type
			if !strings.HasPrefix(contentType, "multipart/form-data; boundary=") {
				t.Errorf("invalid content type: %s", contentType)
			}

			// Extract boundary from content type
			boundary := strings.TrimPrefix(contentType, "multipart/form-data; boundary=")

			// Parse multipart body
			reader := multipart.NewReader(bytes.NewReader(body), boundary)

			// Track found parts
			foundFile := false
			foundFields := make(map[string]bool)

			for {
				part, err := reader.NextPart()
				if err == io.EOF {
					break
				}
				if err != nil {
					t.Fatalf("failed to read multipart part: %v", err)
				}

				formName := part.FormName()

				if formName == tt.fieldName {
					// File part
					foundFile = true

					// Check filename
					if part.FileName() != tt.filename {
						t.Errorf("filename = %q, want %q", part.FileName(), tt.filename)
					}

					// Read and verify file content
					content, err := io.ReadAll(part)
					if err != nil {
						t.Fatalf("failed to read file content: %v", err)
					}

					if string(content) != tt.fileData {
						t.Errorf("file content = %q, want %q", string(content), tt.fileData)
					}
				} else {
					// Field part
					content, err := io.ReadAll(part)
					if err != nil {
						t.Fatalf("failed to read field content: %v", err)
					}

					expectedValue, exists := tt.fields[formName]
					if !exists {
						t.Errorf("unexpected field: %s", formName)
						continue
					}

					if string(content) != expectedValue {
						t.Errorf("field %s = %q, want %q", formName, string(content), expectedValue)
					}

					foundFields[formName] = true
				}
			}

			// Verify file was found
			if !foundFile {
				t.Error("file part not found in multipart body")
			}

			// Verify all fields were found
			for fieldName := range tt.fields {
				if !foundFields[fieldName] {
					t.Errorf("field %s not found in multipart body", fieldName)
				}
			}
		})
	}
}

func TestCreateFormFileErrorCases(t *testing.T) {
	t.Run("nil file reader", func(t *testing.T) {
		_, _, err := CreateFormFile("file", "test.txt", nil, nil)
		if err == nil {
			t.Error("expected error for nil file reader")
		}
	})

	t.Run("empty field name", func(t *testing.T) {
		reader := strings.NewReader("data")
		_, _, err := CreateFormFile("", "test.txt", reader, nil)
		if err == nil {
			t.Error("expected error for empty field name")
		}
	})

	t.Run("empty filename", func(t *testing.T) {
		reader := strings.NewReader("data")
		_, _, err := CreateFormFile("file", "", reader, nil)
		if err == nil {
			t.Error("expected error for empty filename")
		}
	})
}

func TestCreateFormFileLargeData(t *testing.T) {
	// Test with larger data to ensure buffering works correctly
	largeData := strings.Repeat("A", 1024*1024) // 1MB
	reader := strings.NewReader(largeData)

	fields := map[string]string{
		"model": "test-model",
	}

	body, contentType, err := CreateFormFile("file", "large.bin", reader, fields)
	if err != nil {
		t.Fatalf("CreateFormFile() error = %v", err)
	}

	// Verify we got data back
	if len(body) == 0 {
		t.Error("body is empty")
	}

	// Verify content type
	if !strings.HasPrefix(contentType, "multipart/form-data; boundary=") {
		t.Errorf("invalid content type: %s", contentType)
	}

	// Verify the data can be parsed
	boundary := strings.TrimPrefix(contentType, "multipart/form-data; boundary=")
	reader2 := multipart.NewReader(bytes.NewReader(body), boundary)

	foundFile := false
	for {
		part, err := reader2.NextPart()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatalf("failed to read part: %v", err)
		}

		if part.FormName() == "file" {
			foundFile = true
			content, err := io.ReadAll(part)
			if err != nil {
				t.Fatalf("failed to read file: %v", err)
			}
			if len(content) != len(largeData) {
				t.Errorf("file size = %d, want %d", len(content), len(largeData))
			}
		}
	}

	if !foundFile {
		t.Error("file not found in multipart body")
	}
}
