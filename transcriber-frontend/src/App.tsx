import { useState, useCallback } from 'react'
import { Upload, FileAudio, Download, Loader2, CheckCircle, AlertCircle } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Progress } from '@/components/ui/progress'
import { Alert, AlertDescription } from '@/components/ui/alert'
import './App.css'

interface TranscriptionResult {
  file_id: string
  subtitle_file: string
  segments_count: number
  message: string
}

interface UploadResult {
  file_id: string
  filename: string
  size: number
  message: string
}

const SUPPORTED_LANGUAGES = [
  { value: 'auto', label: 'Auto-detect' },
  { value: 'en', label: 'English' },
  { value: 'es', label: 'Spanish' },
  { value: 'fr', label: 'French' },
  { value: 'de', label: 'German' },
  { value: 'it', label: 'Italian' },
  { value: 'pt', label: 'Portuguese' },
  { value: 'ru', label: 'Russian' },
  { value: 'ja', label: 'Japanese' },
  { value: 'ko', label: 'Korean' },
  { value: 'zh', label: 'Chinese' },
  { value: 'ar', label: 'Arabic' },
  { value: 'hi', label: 'Hindi' }
]

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

function App() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [selectedLanguage, setSelectedLanguage] = useState('auto')
  const [uploadResult, setUploadResult] = useState<UploadResult | null>(null)
  const [transcriptionResult, setTranscriptionResult] = useState<TranscriptionResult | null>(null)
  const [isUploading, setIsUploading] = useState(false)
  const [isTranscribing, setIsTranscribing] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [dragActive, setDragActive] = useState(false)

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true)
    } else if (e.type === 'dragleave') {
      setDragActive(false)
    }
  }, [])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0]
      if (isValidFileType(file)) {
        setSelectedFile(file)
        setError(null)
      } else {
        setError('Please select a valid audio or video file (MP3, WAV, AAC, MP4, MOV, AVI, M4A, FLAC)')
      }
    }
  }, [])

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0]
      if (isValidFileType(file)) {
        setSelectedFile(file)
        setError(null)
      } else {
        setError('Please select a valid audio or video file (MP3, WAV, AAC, MP4, MOV, AVI, M4A, FLAC)')
      }
    }
  }

  const isValidFileType = (file: File) => {
    const validTypes = [
      'audio/mpeg', 'audio/wav', 'audio/aac', 'audio/mp4', 'audio/x-m4a', 'audio/flac',
      'video/mp4', 'video/quicktime', 'video/x-msvideo'
    ]
    const validExtensions = ['.mp3', '.wav', '.aac', '.mp4', '.mov', '.avi', '.m4a', '.flac']
    
    return validTypes.includes(file.type) || 
           validExtensions.some(ext => file.name.toLowerCase().endsWith(ext))
  }

  const uploadFile = async () => {
    if (!selectedFile) return

    setIsUploading(true)
    setError(null)

    try {
      const formData = new FormData()
      formData.append('file', selectedFile)

      const response = await fetch(`${API_BASE_URL}/upload`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        throw new Error(`Upload failed: ${response.statusText}`)
      }

      const result: UploadResult = await response.json()
      setUploadResult(result)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed')
    } finally {
      setIsUploading(false)
    }
  }

  const transcribeFile = async () => {
    if (!uploadResult) return

    setIsTranscribing(true)
    setError(null)

    try {
      const formData = new FormData()
      if (selectedLanguage !== 'auto') {
        formData.append('language', selectedLanguage)
      }

      const response = await fetch(`${API_BASE_URL}/transcribe/${uploadResult.file_id}`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        throw new Error(`Transcription failed: ${response.statusText}`)
      }

      const result: TranscriptionResult = await response.json()
      setTranscriptionResult(result)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Transcription failed')
    } finally {
      setIsTranscribing(false)
    }
  }

  const downloadSubtitle = () => {
    if (!transcriptionResult) return
    
    const downloadUrl = `${API_BASE_URL}/download/${transcriptionResult.subtitle_file}`
    const link = document.createElement('a')
    link.href = downloadUrl
    link.download = transcriptionResult.subtitle_file
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
  }

  const resetForm = () => {
    setSelectedFile(null)
    setUploadResult(null)
    setTranscriptionResult(null)
    setError(null)
    setSelectedLanguage('auto')
  }

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-4">
      <div className="max-w-4xl mx-auto">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">
            Multilingual Transcriber
          </h1>
          <p className="text-lg text-gray-600">
            Upload audio or video files to generate subtitles with speaker identification
          </p>
        </div>

        <Card className="mb-6">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <FileAudio className="h-5 w-5" />
              File Upload
            </CardTitle>
            <CardDescription>
              Select an audio or video file to transcribe. Supported formats: MP3, WAV, AAC, MP4, MOV, AVI, M4A, FLAC
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div
              className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
                dragActive
                  ? 'border-blue-500 bg-blue-50'
                  : 'border-gray-300 hover:border-gray-400'
              }`}
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
            >
              {selectedFile ? (
                <div className="space-y-4">
                  <CheckCircle className="h-12 w-12 text-green-500 mx-auto" />
                  <div>
                    <p className="font-medium text-gray-900">{selectedFile.name}</p>
                    <p className="text-sm text-gray-500">{formatFileSize(selectedFile.size)}</p>
                  </div>
                  <Button variant="outline" onClick={() => setSelectedFile(null)}>
                    Choose Different File
                  </Button>
                </div>
              ) : (
                <div className="space-y-4">
                  <Upload className="h-12 w-12 text-gray-400 mx-auto" />
                  <div>
                    <p className="text-lg font-medium text-gray-900">
                      Drop your file here, or click to browse
                    </p>
                    <p className="text-sm text-gray-500">
                      Maximum file size: 100MB
                    </p>
                  </div>
                  <input
                    type="file"
                    accept=".mp3,.wav,.aac,.mp4,.mov,.avi,.m4a,.flac,audio/*,video/*"
                    onChange={handleFileSelect}
                    className="hidden"
                    id="file-upload"
                  />
                  <label htmlFor="file-upload">
                    <Button variant="outline" className="cursor-pointer">
                      Browse Files
                    </Button>
                  </label>
                </div>
              )}
            </div>

            {selectedFile && (
              <div className="mt-6 space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Language (Optional)
                  </label>
                  <Select value={selectedLanguage} onValueChange={setSelectedLanguage}>
                    <SelectTrigger className="w-full">
                      <SelectValue placeholder="Select language" />
                    </SelectTrigger>
                    <SelectContent>
                      {SUPPORTED_LANGUAGES.map((lang) => (
                        <SelectItem key={lang.value} value={lang.value}>
                          {lang.label}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                <Button
                  onClick={uploadFile}
                  disabled={isUploading}
                  className="w-full"
                >
                  {isUploading ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Uploading...
                    </>
                  ) : (
                    <>
                      <Upload className="mr-2 h-4 w-4" />
                      Upload File
                    </>
                  )}
                </Button>
              </div>
            )}
          </CardContent>
        </Card>

        {uploadResult && (
          <Card className="mb-6">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <CheckCircle className="h-5 w-5 text-green-500" />
                File Uploaded Successfully
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="bg-green-50 p-4 rounded-lg">
                  <p className="text-sm text-green-800">
                    <strong>File:</strong> {uploadResult.filename}
                  </p>
                  <p className="text-sm text-green-800">
                    <strong>Size:</strong> {formatFileSize(uploadResult.size)}
                  </p>
                </div>

                <Button
                  onClick={transcribeFile}
                  disabled={isTranscribing}
                  className="w-full"
                >
                  {isTranscribing ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Transcribing... This may take a few minutes
                    </>
                  ) : (
                    'Start Transcription'
                  )}
                </Button>

                {isTranscribing && (
                  <div className="space-y-2">
                    <Progress value={undefined} className="w-full" />
                    <p className="text-sm text-gray-600 text-center">
                      Processing your file... Please wait
                    </p>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        )}

        {transcriptionResult && (
          <Card className="mb-6">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <CheckCircle className="h-5 w-5 text-green-500" />
                Transcription Complete
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="bg-green-50 p-4 rounded-lg">
                  <p className="text-sm text-green-800">
                    <strong>Segments:</strong> {transcriptionResult.segments_count} subtitle segments created
                  </p>
                  <p className="text-sm text-green-800">
                    <strong>File:</strong> {transcriptionResult.subtitle_file}
                  </p>
                </div>

                <div className="flex gap-4">
                  <Button onClick={downloadSubtitle} className="flex-1">
                    <Download className="mr-2 h-4 w-4" />
                    Download Subtitle File
                  </Button>
                  <Button variant="outline" onClick={resetForm}>
                    Process Another File
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        )}

        {error && (
          <Alert className="mb-6" variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        <Card>
          <CardHeader>
            <CardTitle>How it works</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid md:grid-cols-3 gap-6">
              <div className="text-center">
                <div className="bg-blue-100 rounded-full w-12 h-12 flex items-center justify-center mx-auto mb-3">
                  <Upload className="h-6 w-6 text-blue-600" />
                </div>
                <h3 className="font-medium mb-2">1. Upload</h3>
                <p className="text-sm text-gray-600">
                  Upload your audio or video file. We support most common formats.
                </p>
              </div>
              <div className="text-center">
                <div className="bg-blue-100 rounded-full w-12 h-12 flex items-center justify-center mx-auto mb-3">
                  <Loader2 className="h-6 w-6 text-blue-600" />
                </div>
                <h3 className="font-medium mb-2">2. Process</h3>
                <p className="text-sm text-gray-600">
                  Our AI transcribes the audio and identifies different speakers.
                </p>
              </div>
              <div className="text-center">
                <div className="bg-blue-100 rounded-full w-12 h-12 flex items-center justify-center mx-auto mb-3">
                  <Download className="h-6 w-6 text-blue-600" />
                </div>
                <h3 className="font-medium mb-2">3. Download</h3>
                <p className="text-sm text-gray-600">
                  Get your subtitle file with speaker tags and timestamps.
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

export default App
