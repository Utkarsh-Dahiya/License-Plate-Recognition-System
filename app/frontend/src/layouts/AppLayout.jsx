import { Outlet } from 'react-router-dom'
import Sidebar from '../components/Sidebar.jsx'

export default function AppLayout() {
  return (
    <div className="flex h-screen bg-base text-ink">
      <Sidebar />
      <main className="flex-1 overflow-y-auto">
        <Outlet />
      </main>
    </div>
  )
}
