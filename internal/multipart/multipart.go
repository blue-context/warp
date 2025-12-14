// Package multipart provides utilities for creating multipart/form-data encoded requests.
//
// This package uses only the Go standard library (mime/multipart) to avoid
// external dependencies while providing efficient multipart encoding.
package multipart

import (
	"bytes"
	"fmt"
	"io"
	"mime/multipart"
)

// CreateFormFile creates a multipart form with a file field and additional form fields.
//
// The file content is read from the provided io.Reader and written to a buffer.
// Additional form fields are added as specified in the fields map.
//
// Returns the encoded body, the Content-Type header value (including boundary),
// and any error encountered.
//
// Example:
//
//	file, err := os.Open("audio.mp3")
//	if err != nil {
//	    return err
//	}
//	defer file.Close()
//
//	fields := map[string]string{
//	    "model": "whisper-1",
//	    "language": "en",
//	}
//
//	body, contentType, err := multipart.CreateFormFile("file", "audio.mp3", file, fields)
//	if err != nil {
//	    return err
//	}
//
//	req, err := http.NewRequest("POST", url, bytes.NewReader(body))
//	req.Header.Set("Content-Type", contentType)
func CreateFormFile(fieldName, filename string, file io.Reader, fields map[string]string) ([]byte, string, error) {
	if fieldName == "" {
		return nil, "", fmt.Errorf("field name cannot be empty")
	}
	if filename == "" {
		return nil, "", fmt.Errorf("filename cannot be empty")
	}
	if file == nil {
		return nil, "", fmt.Errorf("file reader cannot be nil")
	}

	var buf bytes.Buffer
	writer := multipart.NewWriter(&buf)

	// Add file field
	part, err := writer.CreateFormFile(fieldName, filename)
	if err != nil {
		return nil, "", fmt.Errorf("failed to create form file: %w", err)
	}

	// Copy file content
	if _, err := io.Copy(part, file); err != nil {
		return nil, "", fmt.Errorf("failed to copy file content: %w", err)
	}

	// Add additional fields
	for key, value := range fields {
		if err := writer.WriteField(key, value); err != nil {
			return nil, "", fmt.Errorf("failed to write field %s: %w", key, err)
		}
	}

	// Close writer to finalize multipart message
	if err := writer.Close(); err != nil {
		return nil, "", fmt.Errorf("failed to close multipart writer: %w", err)
	}

	contentType := writer.FormDataContentType()
	return buf.Bytes(), contentType, nil
}
