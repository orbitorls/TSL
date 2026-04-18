import Link from 'next/link'

export default function Navigation() {
  return (
    <nav className="bg-surface-50 border-b border-surface-200 shadow-sm">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center gap-6">
            <Link href="/" className="flex-shrink-0 flex items-center gap-3">
              <div className="w-10 h-10 rounded-full bg-brand-500 flex items-center justify-center text-white font-display">
                T
              </div>
              <span className="text-xl font-semibold text-gray-900 font-display">TSL-51</span>
            </Link>

            <div className="hidden sm:flex ml-6 space-x-6">
              <Link href="/" className="text-sm font-medium text-gray-700 hover:text-brand-600 transition-colors duration-150 font-body">Home</Link>
              <Link href="/translate" className="text-sm font-medium text-gray-700 hover:text-brand-600 transition-colors duration-150 font-body">Translate</Link>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <button className="px-4 py-2 rounded-full bg-brand-500 text-white text-sm font-semibold hover:bg-brand-600 transition">เริ่มแปล</button>
          </div>
        </div>
      </div>
    </nav>
  )
}
