// Package testutil provides testing utilities for the Warp Go SDK.
//
// This package includes assertion helpers, mock implementations, and test fixtures
// to make testing easier throughout the SDK and for users.
//
// Example:
//
//	func TestSomething(t *testing.T) {
//	    assert := testutil.New(t)
//	    assert.NoError(err)
//	    assert.Equal(expected, actual)
//	}
package testutil

import (
	"reflect"
	"strings"
	"testing"
)

// Assert provides simple assertion helpers for testing.
//
// This is a lightweight alternative to testify/assert that requires
// no external dependencies.
//
// Example:
//
//	assert := testutil.New(t)
//	assert.NoError(err)
//	assert.Equal(expected, actual)
type Assert struct {
	t *testing.T
}

// New creates a new Assert instance.
func New(t *testing.T) *Assert {
	return &Assert{t: t}
}

// NoError asserts that err is nil.
func (a *Assert) NoError(err error) {
	a.t.Helper()
	if err != nil {
		a.t.Fatalf("expected no error, got: %v", err)
	}
}

// Error asserts that err is not nil.
func (a *Assert) Error(err error) {
	a.t.Helper()
	if err == nil {
		a.t.Fatal("expected error, got nil")
	}
}

// Equal asserts that expected equals actual using deep equality.
func (a *Assert) Equal(expected, actual any) {
	a.t.Helper()
	if !reflect.DeepEqual(expected, actual) {
		a.t.Fatalf("expected %v, got %v", expected, actual)
	}
}

// NotEqual asserts that expected does not equal actual.
func (a *Assert) NotEqual(expected, actual any) {
	a.t.Helper()
	if reflect.DeepEqual(expected, actual) {
		a.t.Fatalf("expected values to be different, both are %v", expected)
	}
}

// Nil asserts that v is nil.
func (a *Assert) Nil(v any) {
	a.t.Helper()
	if v == nil {
		return
	}
	val := reflect.ValueOf(v)
	kind := val.Kind()
	// Only these kinds can be nil
	if kind == reflect.Ptr || kind == reflect.Interface || kind == reflect.Slice ||
		kind == reflect.Map || kind == reflect.Chan || kind == reflect.Func {
		if !val.IsNil() {
			a.t.Fatalf("expected nil, got %v", v)
		}
	} else {
		a.t.Fatalf("expected nil, got %v", v)
	}
}

// NotNil asserts that v is not nil.
func (a *Assert) NotNil(v any) {
	a.t.Helper()
	if v == nil {
		a.t.Fatal("expected not nil, got nil")
		return
	}
	val := reflect.ValueOf(v)
	kind := val.Kind()
	// Only check IsNil for types that can be nil
	if kind == reflect.Ptr || kind == reflect.Interface || kind == reflect.Slice ||
		kind == reflect.Map || kind == reflect.Chan || kind == reflect.Func {
		if val.IsNil() {
			a.t.Fatal("expected not nil, got nil")
		}
	}
	// For non-nillable types, not being nil (v == nil check above passed) is sufficient
}

// True asserts that v is true.
func (a *Assert) True(v bool) {
	a.t.Helper()
	if !v {
		a.t.Fatal("expected true, got false")
	}
}

// False asserts that v is false.
func (a *Assert) False(v bool) {
	a.t.Helper()
	if v {
		a.t.Fatal("expected false, got true")
	}
}

// Contains asserts that s contains substr.
func (a *Assert) Contains(s, substr string) {
	a.t.Helper()
	if !strings.Contains(s, substr) {
		a.t.Fatalf("expected %q to contain %q", s, substr)
	}
}

// NotContains asserts that s does not contain substr.
func (a *Assert) NotContains(s, substr string) {
	a.t.Helper()
	if strings.Contains(s, substr) {
		a.t.Fatalf("expected %q not to contain %q", s, substr)
	}
}

// Len asserts that v has length n.
func (a *Assert) Len(v any, n int) {
	a.t.Helper()
	val := reflect.ValueOf(v)
	if val.Len() != n {
		a.t.Fatalf("expected length %d, got %d", n, val.Len())
	}
}

// Empty asserts that v is empty (length 0).
func (a *Assert) Empty(v any) {
	a.t.Helper()
	val := reflect.ValueOf(v)
	if val.Len() != 0 {
		a.t.Fatalf("expected empty, got length %d", val.Len())
	}
}

// NotEmpty asserts that v is not empty.
func (a *Assert) NotEmpty(v any) {
	a.t.Helper()
	val := reflect.ValueOf(v)
	if val.Len() == 0 {
		a.t.Fatal("expected not empty, got empty")
	}
}

// Panics asserts that fn panics.
func (a *Assert) Panics(fn func()) {
	a.t.Helper()
	defer func() {
		if r := recover(); r == nil {
			a.t.Fatal("expected panic, but function did not panic")
		}
	}()
	fn()
}

// NotPanics asserts that fn does not panic.
func (a *Assert) NotPanics(fn func()) {
	a.t.Helper()
	defer func() {
		if r := recover(); r != nil {
			a.t.Fatalf("expected no panic, but function panicked with: %v", r)
		}
	}()
	fn()
}
