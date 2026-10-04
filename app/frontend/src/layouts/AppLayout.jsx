import { Outlet } from 'react-router-dom'
import Sidebar from '../components/Sidebar.jsx'

export default function AppLayout() {
  return (
    <div className="flex h-screen overflow-hidden bg-base text-ink">
      {/* Below md the fixed sidebar would squeeze the workspace, so it
          collapses to a slim icon rail; above md the full rail is shown. */}
      <Sidebar />
      {/* min-w-0 lets children shrink instead of forcing horizontal scroll. */}
      <main className="min-w-0 flex-1 overflow-x-hidden overflow-y-auto">
        <Outlet />
      </main>
    </div>
  )
}
