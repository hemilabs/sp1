package sp1

// What the GPU Groth16 prover needs from the circuit's R1CS when the GPU R1CS solver solves it,
// read without parsing the constraints.
//
// gnark's R1CS file (constraint/bn254 marshal.go and constraint/marshal.go at the version go.mod
// pins) is: a uint64 length, the gnark version (three uint64), then the system: a header of four
// uint64 section lengths (levels, instructions, call data, body), the three binary sections, the
// body (CBOR: the wire counts, the public and secret wire names, the commitment info, the debug
// information, ...), and last the coefficient table. The binary sections and the coefficients
// are what solving needs (~1.5 GB for the v6.1.0 circuit, ~19 s of the first proof to parse);
// the GPU solver reads its own form of them (r1cs_solve_plan's output). readR1csMeta seeks past
// them and decodes only the body's counts and commitment info.

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"os"
	"reflect"
	"strings"

	"github.com/blang/semver/v4"
	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/constraint"
	"github.com/fxamacker/cbor/v2"
)

// r1csMeta is what proveGroth16Gpu uses from the R1CS when the GPU R1CS solver solves it.
type r1csMeta struct {
	nbPublic, nbSecret, nbInternal, nbConstraints int
	commitments                                   constraint.Groth16Commitments
}

// nbWires is the number of wires: ONE and the public inputs, the secret inputs, the internal
// wires (gnark's solver's layout).
func (m *r1csMeta) nbWires() int { return m.nbPublic + m.nbSecret + m.nbInternal }

// equal: the same counts and commitment info.
func (m *r1csMeta) equal(o *r1csMeta) bool {
	if m.nbPublic != o.nbPublic || m.nbSecret != o.nbSecret || m.nbInternal != o.nbInternal ||
		m.nbConstraints != o.nbConstraints || len(m.commitments) != len(o.commitments) {
		return false
	}
	for i := range m.commitments {
		if !reflect.DeepEqual(normalizedCommitment(m.commitments[i]), normalizedCommitment(o.commitments[i])) {
			return false
		}
	}
	return true
}

// normalizedCommitment is c with empty slices nil, as decoders differ there.
func normalizedCommitment(c constraint.Groth16Commitment) constraint.Groth16Commitment {
	if len(c.PublicAndCommitmentCommitted) == 0 {
		c.PublicAndCommitmentCommitted = nil
	}
	if len(c.PrivateCommitted) == 0 {
		c.PrivateCommitted = nil
	}
	return c
}

// r1csMetaOf is the metadata of a parsed R1CS.
func r1csMetaOf(r1cs constraint.ConstraintSystem) (*r1csMeta, error) {
	commitments, ok := r1cs.GetCommitments().(constraint.Groth16Commitments)
	if !ok {
		return nil, fmt.Errorf("not a Groth16 R1CS (commitments %T)", r1cs.GetCommitments())
	}
	return &r1csMeta{
		nbPublic:      r1cs.GetNbPublicVariables(),
		nbSecret:      r1cs.GetNbSecretVariables(),
		nbInternal:    r1cs.GetNbInternalVariables(),
		nbConstraints: r1cs.GetNbConstraints(),
		commitments:   commitments,
	}, nil
}

// groth16CommitmentsTag is the CBOR tag gnark writes Groth16Commitments with (constraint's
// getTagSet: tags from 5309735 in the order BlueprintGenericHint, BlueprintGenericR1C,
// Groth16Commitments, ...). TestReadR1csMetaMatchesFullParse checks it against gnark's writer.
const groth16CommitmentsTag = 5309735 + 2

// readR1csMeta reads the metadata of the BN254 R1CS at path, which gnark's WriteTo wrote,
// reading only its body (~1 MB for a stripped circuit, ~0.9 GB of mostly debug information for
// the full v6.1.0 one) and none of its constraints.
func readR1csMeta(path string) (meta *r1csMeta, err error) {
	defer func() {
		if r := recover(); r != nil {
			meta, err = nil, fmt.Errorf("failed to read R1CS metadata %s: %v", path, r)
		}
	}()
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open R1CS %s: %w", path, err)
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		return nil, err
	}
	meta, err = decodeR1csMeta(f, info.Size())
	if err != nil {
		return nil, fmt.Errorf("failed to read R1CS metadata %s: %w", path, err)
	}
	return meta, nil
}

// decodeR1csMeta is readR1csMeta on r, of size bytes.
func decodeR1csMeta(r io.ReadSeeker, size int64) (*r1csMeta, error) {
	// The outer header (constraint/bn254 ReadFrom) and the system's section lengths
	// (constraint.System FromBytes).
	var hdr [8]uint64
	if err := binary.Read(r, binary.LittleEndian, &hdr); err != nil {
		return nil, fmt.Errorf("header: %w", err)
	}
	totalLen, major, minor := hdr[0], hdr[1], hdr[2]
	levelsLen, instructionsLen, calldataLen, bodyLen := hdr[4], hdr[5], hdr[6], hdr[7]
	if major != 0 || minor < 10 {
		return nil, fmt.Errorf("unsupported gnark version %d.%d.%d", major, minor, hdr[3])
	}
	const maxSection = 1 << 40
	for _, l := range []uint64{totalLen, levelsLen, instructionsLen, calldataLen, bodyLen} {
		if l > maxSection {
			return nil, fmt.Errorf("section length %d out of range", l)
		}
	}
	systemLen := 32 + levelsLen + instructionsLen + calldataLen + bodyLen
	if systemLen > totalLen || int64(32+totalLen) != size {
		return nil, fmt.Errorf("sections of %d bytes in a system of %d, a file of %d", systemLen, totalLen, size)
	}
	bodyStart := int64(64 + levelsLen + instructionsLen + calldataLen)
	if _, err := r.Seek(bodyStart, io.SeekStart); err != nil {
		return nil, err
	}

	fields, err := readCborMapFields(io.LimitReader(r, int64(bodyLen)), int64(bodyLen), r1csMetaFields)
	if err != nil {
		return nil, fmt.Errorf("body: %w", err)
	}
	dm, err := cbor.DecOptions{MaxArrayElements: 2147483647, MaxMapPairs: 2147483647}.DecMode()
	if err != nil {
		return nil, err
	}
	var body struct {
		GnarkVersion        string
		ScalarField         string
		Type                constraint.SystemType
		NbConstraints       int
		NbInternalVariables int
		Public, Secret      []string
		CommitmentInfo      cbor.RawTag
	}
	for name, dst := range map[string]any{
		"GnarkVersion": &body.GnarkVersion, "ScalarField": &body.ScalarField, "Type": &body.Type,
		"NbConstraints": &body.NbConstraints, "NbInternalVariables": &body.NbInternalVariables,
		"Public": &body.Public, "Secret": &body.Secret, "CommitmentInfo": &body.CommitmentInfo,
	} {
		raw, ok := fields[name]
		if !ok {
			return nil, fmt.Errorf("body has no %s", name)
		}
		if err := dm.Unmarshal(raw, dst); err != nil {
			return nil, fmt.Errorf("body %s: %w", name, err)
		}
	}

	if _, err := semver.Parse(body.GnarkVersion); err != nil {
		return nil, fmt.Errorf("gnark version %q: %w", body.GnarkVersion, err)
	}
	if body.ScalarField != ecc.BN254.ScalarField().Text(16) {
		return nil, fmt.Errorf("scalar field %s is not BN254's", body.ScalarField)
	}
	if body.Type != constraint.SystemR1CS {
		return nil, fmt.Errorf("system type %d is not an R1CS", body.Type)
	}
	if body.CommitmentInfo.Number != groth16CommitmentsTag {
		return nil, fmt.Errorf("commitment info tagged %d, not %d", body.CommitmentInfo.Number, groth16CommitmentsTag)
	}
	var commitments constraint.Groth16Commitments
	if err := dm.Unmarshal(body.CommitmentInfo.Content, &commitments); err != nil {
		return nil, fmt.Errorf("commitment info: %w", err)
	}
	if commitments == nil {
		commitments = constraint.Groth16Commitments{}
	}
	if body.NbConstraints < 0 || body.NbInternalVariables < 0 || len(body.Public) == 0 {
		return nil, errors.New("negative counts or no public wire (ONE)")
	}
	return &r1csMeta{
		nbPublic:      len(body.Public),
		nbSecret:      len(body.Secret),
		nbInternal:    body.NbInternalVariables,
		nbConstraints: body.NbConstraints,
		commitments:   commitments,
	}, nil
}

// r1csMetaFields are the fields of the system's CBOR body (constraint.System, keyed by field
// name) readR1csMeta decodes.
var r1csMetaFields = map[string]bool{
	"GnarkVersion": true, "ScalarField": true, "Type": true, "NbConstraints": true,
	"NbInternalVariables": true, "Public": true, "Secret": true, "CommitmentInfo": true,
}

// readCborMapFields reads the CBOR map with text keys that r holds (size bytes, all of it) and
// returns the encoded values of the keys in want, skipping the others without decoding them:
// fxamacker/cbor's decoder took 5.3 s to skip the 0.9 GB of debug information in the v6.1.0
// circuit's body. Indefinite lengths, which gnark's deterministic encoder does not write, are an
// error, and so is a key in want named twice or a key naming one in another case: gnark's decoder
// takes a field from the first key that names it, ignoring case, which need not be the value
// read here. gnark's encoder writes each field once, under its name.
func readCborMapFields(r io.Reader, size int64, want map[string]bool) (map[string][]byte, error) {
	s := &cborSkipper{r: r, buf: make([]byte, 0, 4<<20)}
	major, n, err := s.head()
	if err != nil {
		return nil, err
	}
	if major != 5 {
		return nil, fmt.Errorf("not a CBOR map (major type %d)", major)
	}
	fields := make(map[string][]byte)
	for i := uint64(0); i < n; i++ {
		major, l, err := s.head()
		if err != nil {
			return nil, err
		}
		if major != 3 || l > 1<<16 {
			return nil, fmt.Errorf("map key %d: major type %d, length %d", i, major, l)
		}
		if err := s.ensure(int(l)); err != nil {
			return nil, err
		}
		key := string(s.buf[s.pos : s.pos+int(l)])
		s.pos += int(l)
		for name := range want {
			if _, seen := fields[name]; strings.EqualFold(key, name) && (key != name || seen) {
				return nil, fmt.Errorf("map key %s: %s named twice or in another case", key, name)
			}
		}
		if want[key] {
			s.capture = new(bytes.Buffer)
			s.captureFrom = s.pos
		}
		if err := s.skip(); err != nil {
			return nil, fmt.Errorf("value of %s: %w", key, err)
		}
		if s.capture != nil {
			s.capture.Write(s.buf[s.captureFrom:s.pos])
			fields[key] = s.capture.Bytes()
			s.capture = nil
		}
	}
	if s.offset+int64(s.pos) != size {
		return nil, fmt.Errorf("map of %d bytes in a body of %d", s.offset+int64(s.pos), size)
	}
	return fields, nil
}

// cborSkipper walks CBOR data items in buf[pos:], which it refills from r; with capture set, it
// copies what it walks from buf[captureFrom:] there before buf is refilled.
type cborSkipper struct {
	r           io.Reader
	buf         []byte
	pos         int
	offset      int64 // of buf[0] in the data
	eof         bool
	capture     *bytes.Buffer
	captureFrom int
}

// ensure makes n bytes available at buf[pos:], n at most the buffer's capacity.
func (s *cborSkipper) ensure(n int) error {
	if len(s.buf)-s.pos >= n {
		return nil
	}
	if n > cap(s.buf) {
		return fmt.Errorf("item of %d bytes", n)
	}
	if s.capture != nil {
		s.capture.Write(s.buf[s.captureFrom:s.pos])
		s.captureFrom = 0
	}
	s.offset += int64(s.pos)
	s.buf = s.buf[:copy(s.buf[:cap(s.buf)], s.buf[s.pos:])]
	s.pos = 0
	for len(s.buf) < n {
		if s.eof {
			return io.ErrUnexpectedEOF
		}
		m, err := s.r.Read(s.buf[len(s.buf):cap(s.buf)])
		s.buf = s.buf[:len(s.buf)+m]
		if err == io.EOF {
			s.eof = true
		} else if err != nil {
			return err
		}
	}
	return nil
}

// head reads an item's initial byte and argument: its major type, and its value, length or
// number of elements.
func (s *cborSkipper) head() (major byte, arg uint64, err error) {
	if len(s.buf)-s.pos < 9 {
		// At most 9 bytes; fewer at the end of the data.
		if err := s.ensure(1); err != nil {
			return 0, 0, err
		}
		if err := s.ensure(1 + cborArgLen(s.buf[s.pos]&31)); err != nil {
			return 0, 0, err
		}
	}
	b := s.buf[s.pos]
	major, ai := b>>5, b&31
	switch {
	case ai < 24:
		s.pos++
		return major, uint64(ai), nil
	case ai == 24:
		arg = uint64(s.buf[s.pos+1])
	case ai == 25:
		arg = uint64(binary.BigEndian.Uint16(s.buf[s.pos+1:]))
	case ai == 26:
		arg = uint64(binary.BigEndian.Uint32(s.buf[s.pos+1:]))
	case ai == 27:
		arg = binary.BigEndian.Uint64(s.buf[s.pos+1:])
	default:
		return 0, 0, fmt.Errorf("unsupported CBOR initial byte %#x", b)
	}
	s.pos += 1 + cborArgLen(ai)
	return major, arg, nil
}

// cborArgLen is the number of argument bytes after an initial byte with additional info ai.
func cborArgLen(ai byte) int {
	if ai < 24 || ai > 27 {
		return 0
	}
	return 1 << (ai - 24)
}

// discard skips n bytes of a byte or text string.
func (s *cborSkipper) discard(n uint64) error {
	for n > 0 {
		if s.pos == len(s.buf) {
			if err := s.ensure(1); err != nil {
				return err
			}
		}
		k := uint64(len(s.buf) - s.pos)
		if k > n {
			k = n
		}
		s.pos += int(k)
		n -= k
	}
	return nil
}

// skip reads one data item, with what it contains.
func (s *cborSkipper) skip() error {
	for pending := uint64(1); pending > 0; pending-- {
		major, arg, err := s.head()
		if err != nil {
			return err
		}
		if arg > 1<<40 && major >= 2 && major <= 5 {
			return fmt.Errorf("CBOR item of %d bytes or elements", arg)
		}
		switch major {
		case 0, 1, 7: // integers, simple values and floats: all in the head
		case 2, 3:
			if err := s.discard(arg); err != nil {
				return err
			}
		case 4:
			pending += arg
		case 5:
			pending += 2 * arg
		case 6: // a tag: then its content
			pending++
		}
		if pending > 1<<40 {
			return errors.New("too many nested elements")
		}
	}
	return nil
}
