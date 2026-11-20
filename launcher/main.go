package main

import (
	"bufio"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"
)

const vitePort = 5173

func main() {
	fmt.Println("====================================================")
	fmt.Println("  OpenASL — Automatic Setup & Launcher (Temporary)")
	fmt.Println("====================================================")

	rootDir, err := findRepoRoot()
	if err != nil {
		fatalPause(err.Error())
	}
	fmt.Printf("[Setup] Repo directory: %s\n", rootDir)

	// 1. Check Node / npm
	npmBin := "npm"
	if runtime.GOOS == "windows" {
		npmBin = "npm.cmd"
	}
	if _, err := exec.LookPath(npmBin); err != nil {
		fatalPause("Node.js / npm was not found on PATH.\nInstall Node.js LTS from https://nodejs.org and re-run OpenASL-Start.exe.")
	}

	// 2. Setup Frontend (npm install if node_modules missing)
	nodeModules := filepath.Join(rootDir, "node_modules")
	if _, err := os.Stat(nodeModules); os.IsNotExist(err) {
		fmt.Println("[Frontend] Installing npm dependencies (first run)...")
		if err := runForeground(rootDir, npmBin, "install"); err != nil {
			fatalPause(fmt.Sprintf("npm install failed: %v", err))
		}
	} else {
		fmt.Println("[Frontend] node_modules found.")
	}

	// 3. Setup Python Virtual Camera Backend (bridge/.venv)
	bridgePython, bridgeReady := setupPythonBridge(rootDir)

	// 4. On Windows, check if zero-OBS UnityCapture virtual cam DLL is installed
	if runtime.GOOS == "windows" && bridgeReady {
		ensureWindowsVirtualCamDriver(rootDir)
	}

	// 5. Start Backend & Frontend concurrently
	var wg sync.WaitGroup
	var mu sync.Mutex
	var children []*exec.Cmd

	addChild := func(cmd *exec.Cmd) {
		mu.Lock()
		children = append(children, cmd)
		mu.Unlock()
	}

	if bridgeReady {
		bridgeScript := filepath.Join(rootDir, "bridge", "virtualcam_bridge.py")
		bridgeCmd := exec.Command(bridgePython, "-u", bridgeScript)
		bridgeCmd.Dir = rootDir
		if err := startPrefixed("[Bridge]", bridgeCmd, &wg); err != nil {
			fmt.Printf("[Bridge] Warning: could not start virtual camera bridge: %v\n", err)
		} else {
			addChild(bridgeCmd)
		}
	} else {
		fmt.Println("[Bridge] Skipping virtual camera bridge (Python setup unavailable). Frontend will still run!")
	}

	frontendCmd := exec.Command(npmBin, "run", "dev", "--", "--port", strconv.Itoa(vitePort))
	frontendCmd.Dir = rootDir
	if err := startPrefixed("[WebAI]", frontendCmd, &wg); err != nil {
		killAll(children)
		fatalPause(fmt.Sprintf("Failed to start Vite dev server: %v", err))
	}
	addChild(frontendCmd)

	// 6. Wait for Vite HTTP server and launch browser automatically
	appURL := fmt.Sprintf("http://localhost:%d", vitePort)
	go func() {
		if waitForHTTP(appURL, 30*time.Second) {
			fmt.Printf("\n[OpenASL] Ready! Opening %s ...\n", appURL)
			fmt.Println("[OpenASL] Press Ctrl+C (or close this window) to stop both servers.")
			openBrowser(appURL)
		} else {
			fmt.Printf("\n[OpenASL] Open %s in Chrome or Edge once Vite finishes starting.\n", appURL)
		}
	}()

	// 7. Wait for Ctrl+C / termination signal and cleanly tear down child trees
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, os.Interrupt, syscall.SIGTERM)

	doneCh := make(chan struct{})
	go func() {
		_ = frontendCmd.Wait()
		close(doneCh)
	}()

	select {
	case <-sigCh:
		fmt.Println("\n[OpenASL] Shutting down frontend & virtual camera bridge...")
	case <-doneCh:
		fmt.Println("\n[OpenASL] Frontend exited. Shutting down bridge...")
	}

	killAll(children)
	wg.Wait()
	fmt.Println("[OpenASL] Stopped cleanly.")
}

func findRepoRoot() (string, error) {
	candidates := []string{}
	if exePath, err := os.Executable(); err == nil {
		candidates = append(candidates, filepath.Dir(exePath))
	}
	if cwd, err := os.Getwd(); err == nil {
		candidates = append(candidates, cwd)
	}
	for _, dir := range candidates {
		pkg := filepath.Join(dir, "package.json")
		if _, err := os.Stat(pkg); err == nil {
			return dir, nil
		}
	}
	return "", fmt.Errorf("could not find package.json next to OpenASL-Start.exe or in current folder.\nPlace OpenASL-Start.exe inside your OpenASL repo folder and run it again")
}

func findSystemPython() (string, []string, bool) {
	if runtime.GOOS == "windows" {
		if p, err := exec.LookPath("python"); err == nil {
			return p, nil, true
		}
		if p, err := exec.LookPath("py"); err == nil {
			return p, []string{"-3"}, true
		}
		return "", nil, false
	}
	if p, err := exec.LookPath("python3"); err == nil {
		return p, nil, true
	}
	if p, err := exec.LookPath("python"); err == nil {
		return p, nil, true
	}
	return "", nil, false
}

func setupPythonBridge(rootDir string) (string, bool) {
	venvDir := filepath.Join(rootDir, "bridge", ".venv")
	var venvPython string
	if runtime.GOOS == "windows" {
		venvPython = filepath.Join(venvDir, "Scripts", "python.exe")
	} else {
		venvPython = filepath.Join(venvDir, "bin", "python")
	}

	if _, err := os.Stat(venvPython); os.IsNotExist(err) {
		sysPy, pyPrefixArgs, ok := findSystemPython()
		if !ok {
			fmt.Println("[Bridge] Python 3 not found on PATH. Install Python 3 to enable Virtual Camera output.")
			return "", false
		}
		fmt.Println("[Bridge] Creating isolated Python virtual environment in bridge/.venv ...")
		args := append(pyPrefixArgs, "-m", "venv", venvDir)
		if err := runForeground(rootDir, sysPy, args...); err != nil {
			fmt.Printf("[Bridge] Failed to create venv: %v\n", err)
			return "", false
		}
	}

	// Quick import check to skip pip install on subsequent launches
	checkCmd := exec.Command(venvPython, "-c", "import pyvirtualcam, websockets, cv2, numpy")
	if err := checkCmd.Run(); err != nil {
		fmt.Println("[Bridge] Installing virtual camera Python packages (first run)...")
		reqFile := filepath.Join(rootDir, "bridge", "requirements.txt")
		if err := runForeground(rootDir, venvPython, "-m", "pip", "install", "--quiet", "-r", reqFile); err != nil {
			fmt.Printf("[Bridge] pip install failed: %v\n", err)
			return "", false
		}
	} else {
		fmt.Println("[Bridge] Python virtual camera dependencies ready.")
	}

	return venvPython, true
}

func ensureWindowsVirtualCamDriver(rootDir string) {
	progFiles := os.Getenv("ProgramFiles")
	if progFiles == "" {
		progFiles = `C:\Program Files`
	}
	dllPath := filepath.Join(progFiles, "OpenASL", "driver", "UnityCaptureFilter64.dll")
	if _, err := os.Stat(dllPath); err == nil {
		fmt.Println("[Driver] OpenASL zero-OBS virtual camera driver detected.")
		return
	}

	installScript := filepath.Join(rootDir, "bridge", "install_windows_vcam.ps1")
	if _, err := os.Stat(installScript); os.IsNotExist(err) {
		return
	}

	fmt.Println("\n[Driver] Zero-OBS Virtual Webcam driver (UnityCaptureFilter64.dll) is not installed yet.")
	fmt.Print("[Driver] Install it now? (Requires Windows Admin UAC prompt) [Y/n]: ")
	reader := bufio.NewReader(os.Stdin)
	line, _ := reader.ReadString('\n')
	answer := strings.ToLower(strings.TrimSpace(line))
	if answer == "" || answer == "y" || answer == "yes" {
		psArgs := fmt.Sprintf(
			"Start-Process powershell -Verb RunAs -Wait -ArgumentList '-ExecutionPolicy','Bypass','-File','\"%s\"'",
			installScript,
		)
		cmd := exec.Command("powershell", "-NoProfile", "-Command", psArgs)
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
		if err := cmd.Run(); err != nil {
			fmt.Printf("[Driver] Warning: driver installer did not complete (%v). You can run bridge\\install_windows_vcam.ps1 later.\n", err)
		}
	}
}

func runForeground(dir string, name string, args ...string) error {
	cmd := exec.Command(name, args...)
	cmd.Dir = dir
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.Stdin = os.Stdin
	return cmd.Run()
}

func startPrefixed(prefix string, cmd *exec.Cmd, wg *sync.WaitGroup) error {
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return err
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		return err
	}
	if err := cmd.Start(); err != nil {
		return err
	}
	pump := func(r io.Reader) {
		defer wg.Done()
		sc := bufio.NewScanner(r)
		buf := make([]byte, 0, 64*1024)
		sc.Buffer(buf, 512*1024)
		for sc.Scan() {
			fmt.Printf("%s %s\n", prefix, sc.Text())
		}
	}
	wg.Add(2)
	go pump(stdout)
	go pump(stderr)
	return nil
}

func waitForHTTP(url string, timeout time.Duration) bool {
	deadline := time.Now().Add(timeout)
	client := &http.Client{Timeout: 800 * time.Millisecond}
	for time.Now().Before(deadline) {
		resp, err := client.Get(url)
		if err == nil {
			_ = resp.Body.Close()
			if resp.StatusCode < 500 {
				return true
			}
		}
		time.Sleep(350 * time.Millisecond)
	}
	return false
}

func openBrowser(url string) {
	var cmd *exec.Cmd
	switch runtime.GOOS {
	case "windows":
		cmd = exec.Command("rundll32", "url.dll,FileProtocolHandler", url)
	case "darwin":
		cmd = exec.Command("open", url)
	default:
		cmd = exec.Command("xdg-open", url)
	}
	_ = cmd.Start()
}

func killAll(cmds []*exec.Cmd) {
	for _, cmd := range cmds {
		if cmd == nil || cmd.Process == nil {
			continue
		}
		if runtime.GOOS == "windows" {
			// Kill entire process tree (e.g. cmd.exe -> node.exe / vite)
			_ = exec.Command("taskkill", "/F", "/T", "/PID", strconv.Itoa(cmd.Process.Pid)).Run()
		} else {
			_ = cmd.Process.Signal(syscall.SIGTERM)
		}
	}
}

func fatalPause(msg string) {
	fmt.Fprintf(os.Stderr, "\n[Error] %s\n", msg)
	if runtime.GOOS == "windows" {
		fmt.Print("\nPress Enter to close...")
		_, _ = bufio.NewReader(os.Stdin).ReadString('\n')
	}
	os.Exit(1)
}
