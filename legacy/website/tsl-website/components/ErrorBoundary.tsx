'use client'

import { useState, useEffect, ReactNode } from 'react'

interface ErrorBoundaryProps {
  children: ReactNode
  fallback?: ReactNode
  onReset?: () => void
  onError?: (error: Error, errorInfo: React.ErrorInfo) => void
}

interface ErrorState {
  hasError: boolean
  error: Error | null
}

export default function ErrorBoundary({
  children,
  fallback,
  onReset,
  onError
}: ErrorBoundaryProps) {
  const [errorState, setErrorState] = useState<ErrorState>({ hasError: false, error: null })

  useEffect(() => {
    const handleError = (event: ErrorEvent) => {
      const error = new Error(event.message)
      error.stack = event.error?.stack
      setErrorState({ hasError: true, error })
      onError?.(error, { componentStack: event.error?.stack || '' })
    }

    const handleUnhandledRejection = (event: PromiseRejectionEvent) => {
      const error = event.reason instanceof Error ? event.reason : new Error(String(event.reason))
      setErrorState({ hasError: true, error })
      onError?.(error, { componentStack: '' })
    }

    window.addEventListener('error', handleError)
    window.addEventListener('unhandledrejection', handleUnhandledRejection)

    return () => {
      window.removeEventListener('error', handleError)
      window.removeEventListener('unhandledrejection', handleUnhandledRejection)
    }
  }, [onError])

  const resetErrorBoundary = () => {
    setErrorState({ hasError: false, error: null })
    onReset?.()
  }

  if (errorState.hasError) {
    if (fallback) {
      return fallback
    }

    return (
      <div className="min-h-[200px] flex flex-col items-center justify-center p-6 bg-red-50 rounded-lg border border-red-200">
        <h2 className="text-lg font-semibold text-red-800 mb-2">Something went wrong</h2>
        <p className="text-sm text-red-600 mb-4">
          {errorState.error?.message || 'An unexpected error occurred'}
        </p>
        <button
          onClick={resetErrorBoundary}
          className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors"
        >
          Try again
        </button>
      </div>
    )
  }

  return children
}