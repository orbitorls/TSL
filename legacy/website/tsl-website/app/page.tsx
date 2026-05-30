import Link from 'next/link'
import Navigation from '@/components/Navigation'

export default function Home() {
  return (
    <div className="min-h-screen bg-surface-50">
      <Navigation />
      <main className="max-w-7xl mx-auto py-16 sm:py-24 px-4 sm:px-6 lg:px-8">
        <div className="text-center animate-fade-in">
          <h1 className="text-5xl font-bold tracking-tight text-gray-900 sm:text-7xl font-display">
            Thai Sign Language
            <span className="block text-brand-600">Translation</span>
          </h1>
          <p className="mt-6 text-lg leading-8 text-gray-600 max-w-2xl mx-auto font-body">
            แปลงภาษามือไทยเป็นข้อความแบบเรียลไทม์ด้วย AI
            <br />
            ช่วยให้การสื่อสารกับผู้พิการทางการได้ยินเป็นไปได้ง่ายขึ้น
          </p>
          <div className="mt-10 flex items-center justify-center gap-x-6">
            <Link
              href="/translate"
              className="rounded-full bg-brand-500 px-8 py-3 text-sm font-semibold text-white shadow-md hover:bg-brand-400 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brand-500 transition-all duration-200 hover:scale-105"
            >
              เริ่มแปลภาษามือ
            </Link>
          </div>
        </div>
        
        {/* Features */}
        <div className="mt-24 grid grid-cols-1 gap-8 sm:grid-cols-3 animate-slide-up">
          <div className="text-center p-6 rounded-2xl bg-surface-100 hover:bg-surface-200 transition-colors duration-200">
            <div className="text-4xl mb-3">📹</div>
            <h3 className="text-lg font-semibold font-display">Real-time</h3>
            <p className="text-gray-600 mt-2 font-body">แปลทันที ขณะใช้กล้อง</p>
          </div>
          <div className="text-center p-6 rounded-2xl bg-surface-100 hover:bg-surface-200 transition-colors duration-200">
            <div className="text-4xl mb-3">🤖</div>
            <h3 className="text-lg font-semibold font-display">AI Powered</h3>
            <p className="text-gray-600 mt-2 font-body">ใช้ Deep Learning Model</p>
          </div>
          <div className="text-center p-6 rounded-2xl bg-surface-100 hover:bg-surface-200 transition-colors duration-200">
            <div className="text-4xl mb-3">🌐</div>
            <h3 className="text-lg font-semibold font-display">51 คำ</h3>
            <p className="text-gray-600 mt-2 font-body">รองรับคำศัพท์พื้นฐาน</p>
          </div>
        </div>
      </main>
    </div>
  )
}